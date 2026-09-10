# Understanding V0: from a tactical question to a tested baseline

V0 is the first version of the CS2 Tactical Intelligence research project: an Early-Round Site Outcome Forecaster. It predicts the eventual recorded plant outcome of an eligible Mirage round from information available before the plant or round end. It returns three probabilities: A plant, B plant, and no plant.

The important result is more than a winning model name. V0 established a way to investigate data, challenge assumptions, compare representations, evaluate uncertainty, and let observed failure modes shape the next question. This chapter explains that reasoning in order. Project-specific numbers come from the frozen local evaluation artifacts; explanatory examples are labeled as illustrations.

## 01 · The question comes before the architecture

The long-term ambition is a system that can observe a match, recognize tactical behavior, forecast what happens next, retrieve similar historical situations, model an opponent, and eventually evaluate possible responses. Each of those capabilities asks a different question. A model that forecasts a plant site does not automatically recognize a fake, understand a team's intention, or know which counter-strategy would win.

The early temptation was to build a generic GameState object immediately. The problem was that its fields would have reflected assumptions about both the data and the future model. The design therefore stepped back: first decide what must be predicted, inspect real source contracts, and only then decide which state representation is needed.

An initial target such as “recognize an execute” creates a labeling problem. Two analysts may disagree about exactly when an execute begins or whether a round is a default, split, or fake. A recorded bomb plant provides a more objective starting point: there is an event, a time, and a site that can be audited. The no-plant class keeps rounds in which neither site receives a plant inside the task.

This choice deliberately leaves tactical meaning incomplete. A team can initially intend to attack A and ultimately plant B. It can prepare a good attack and lose the duels before planting. V0 learns associations with the eventual outcome, not privileged knowledge of intent. Its first success criterion is whether pre-outcome state improves probabilistic forecasting beyond simple baselines.

The work proceeds as a controlled chain:

```text
Define the target → inspect sources → construct labels and observations
→ validate features → compare models → analyze errors and uncertainty
→ freeze the evidence → propose the next experiment
```

## 02 · Exactly what one prediction means

The scope is Mirage, viewed through the attacking T side of each round. The input is full observer-derived state, including CT information where the selected feature set requires it. “T-side forecasting” describes the round's attacking outcome; it does not mean a normal T player could know every input.

For state history available through observation time t, the target is:

```text
P(Y | S[0:t]), where Y ∈ {A_PLANT, B_PLANT, NO_PLANT}

Implemented V0: history → engineered feature vector x(t) → model → probabilities
```

The notation permits a history of observations. The implementation is a tabular model with current-state features plus a short movement summary. It is not a neural sequence model reading the entire round. That distinction matters when comparing V0 to future GRU or TCN candidates.

An illustrative output is P(A) = 0.55, P(B) = 0.20, and P(NO) = 0.25. The three values sum to one. A is the most likely single class, but 45% of the probability remains on other outcomes. If the actual outcome is A, a model that assigned A 0.55 receives a different probability-quality score from one that assigned A 0.99, even though both classify the round as A.

A plant label also does not describe the round winner. A T-side plant followed by a CT defuse is still a plant observation. Conversely, a round can end without a plant for different reasons. Those reasons are compressed into NO_PLANT rather than separately labeled in V0. That deliberately broad class will later become central to the error analysis.

V0 does not currently generate tactical recommendations, infer concealed intentions, evaluate counterfactual actions, or provide demonstrated live predictions. It is a frozen offline research baseline for one precisely defined task.

## 03 · Time, eligibility, and the information boundary

Four observation horizons were fixed: 10, 20, 30, and 40 seconds after freeze_end. Freeze end is the point after the pre-round freeze period. Starting the clock there makes the horizons describe playable round time rather than time spent waiting before the round starts.

```text
target_tick = freeze_end + horizon_seconds × demo_tickrate

Eligible only when:
  target_tick < round_end_tick
  and (there is no plant event, or target_tick < plant_tick)
```

The strict inequality is intentional. A snapshot at the plant event is already too late to forecast whether and where the plant will happen. Similarly, an ended round is not an ongoing early-round forecasting opportunity. Features must be constructed from state at or before the cutoff, never from a later convenient snapshot or future bomb position. The consolidated offline builder requires an exact player snapshot at the target tick and at the one-second lag tick; it raises an error if either is missing. It does not silently substitute a future snapshot.

Consider an illustrative round that plants at 27 seconds and ends at 43 seconds. Its 10-second and 20-second observations are eligible. Its 30-second and 40-second observations are excluded because the plant has already happened. A different round ending without a plant at 18 seconds contributes only a 10-second observation. A round lasting beyond 40 seconds without a plant can contribute all four.

The future event is allowed to define the training label. Supervised learning requires us to look back later and identify what happened. It is not allowed to define a predictor that would have been unknown at t. The distinction is between future information used as the answer and future information leaking into the question.

There are two separate audits: within each observation, check the time boundary; across training and evaluation, check the match boundary. Passing one does not make the other unnecessary. Reconstructing the correct label is also a separate responsibility: an accurately timed feature cannot repair incorrect ground truth.

## 04 · The dataset and why its size needs context

The frozen development dataset contains 20 Mirage demos, 444 parsed rounds, and 443 rounds with valid timing. One partial opening round was excluded. These yield 1,268 eligible observations with zero missing feature values in the constructed dataset.

| Horizon after freeze end | Eligible observations | Share of all observations |
| --- | ---: | ---: |
| 10 seconds | 442 | 34.9% |
| 20 seconds | 373 | 29.4% |
| 30 seconds | 270 | 21.3% |
| 40 seconds | 183 | 14.4% |
| Total | 1,268 | 100% |

These are 1,268 snapshots, not 1,268 independent matches. Observations from the same round are related, and rounds from the same match share teams, opponents, and match conditions. More rows can help a model fit, but they do not automatically provide the same diversity as more independent matches.

The overall label counts derived from the confusion matrix are 422 A_PLANT, 223 B_PLANT, and 623 NO_PLANT observations. The imbalance explains why a simplistic majority-class choice can achieve about 49.1% accuracy while doing poorly on the two site classes. It also explains why macro F1 and per-class recall are useful companions to overall accuracy.

Later horizons contain a selected population: rounds that have survived without an earlier plant or round end. A better metric at 40 seconds may reflect additional information, a different set of rounds, or both. It cannot be interpreted as a causal improvement obtained by waiting on an identical cohort. An identical-cohort study would require a separately defined comparison.

“No missing values” is a construction check, not proof that the data is free of semantic error. A field can be present, finite, and consistently wrong. That is exactly why the project inspected labels and bomb-state logic before treating the dataset as trustworthy.

## 05 · Two source investigations that changed the design

The first investigation concerned plant labels. The obvious shortcut was to trust the parser's round-level bomb_site field. In the initial demo audit, that field disagreed with the actual plant events, reporting B for planted rounds whose event identified A. This was a problem in the observed source path, not a reason to assume every parser version or every demo has the same failure.

The resulting label policy uses bomb plant events directly: BombsiteA maps to A_PLANT; BombsiteB maps to B_PLANT; no recorded plant maps to NO_PLANT. This replaces an unverified convenience field with an auditable event-based derivation. The absence of a plant still depends on successful event parsing, so event completeness remains part of the source contract.

The second investigation concerned current bomb position. “Use the latest bomb event” sounds sensible until a bomb is dropped and then picked up. The last drop coordinate can remain in the event history while the current carrier has already moved elsewhere. A stale bomb position would contaminate several spatial and motion features at once.

The adopted policy prioritizes current inventory: if exactly one T currently carries C4, use that player's current position. Otherwise, reconstruct a dropped bomb from a valid prior drop event. The reconstruction has to respect the observation time and round. Ambiguous or absent evidence needs an audit path rather than an invented location.

On the original 49 eligible observations, the bomb-state audit reported 35 carried, 14 dropped, and zero suspicious states. That is evidence that the implemented rule handled the audited observations; it is not a guarantee about every future demo or every exceptional game event.

Both investigations teach the same engineering lesson: validate what a field means before optimizing a model that depends on it. Better machine learning cannot reliably compensate for mislabeled outcomes or a bomb position reconstructed from the wrong event.

## 06 · From raw state to a feature contract

Raw demo tables contain more fields than V0 needs. The model receives an explicit list, which prevents labels, identifiers, and convenient future-derived fields from entering accidentally. Match and round identifiers remain useful for joining, tracing, and splitting data; they are not automatically tactical predictors.

The selected XGB-A5 model uses 37 inputs:

| Family | Inputs | Purpose |
| --- | ---: | --- |
| Observation horizon | 1 | Identify whether this is the 10, 20, 30, or 40 second observation |
| Offensive geometry | 12 | Describe T formation and bomb location |
| Motion | 4 | Summarize movement over approximately one second |
| Combat | 9 | Describe surviving players, health, and armor on both sides |
| Defense and interaction | 11 | Describe CT formation and T–CT spatial relationships |
| Total selected | 37 | XGB-A5 feature contract |
| Economy, tested separately | 3 | Current equipment-value totals and difference; excluded from A5 |

The family names describe information, while the A0–A5 labels describe experiments. A1 includes horizon plus offensive geometry. A2 adds motion. A3 adds combat. A4 tests economy on top of A3. A5 tests defense on top of A3, not on top of A4. This branching structure is essential for interpreting the experiments.

The longer-term architecture keeps raw-source adapters outside a shared representation and feature builder. Historical demos, replay, and an observer-compatible live source should ultimately produce equivalent feature meanings. In V0, position-derived movement was chosen partly because it can be reconstructed from position history instead of depending on a parser-specific velocity field. This is a design for future parity, not evidence that the live adapter already achieves it.

## 07 · Geometry: where the team is and what shape it makes

A team centroid averages alive-player coordinates. It represents an approximate center, but two radically different formations can share it. In an illustrative one-dimensional example, players at positions −10 and +10 have the same center as players at −1 and +1. The first pair is much more spread out. The model therefore needs more than an average position.

The offensive family contains T centroid x/y/z, mean XY distance to that centroid (stretch), x and y ranges, mean pairwise XY distance, and convex hull area. It also contains bomb x/y/z and bomb-to-T-centroid distance. Together these describe approximate location, dispersion, extent, and whether the bomb is traveling with the formation.

```text
centroid = average of alive-player coordinates
stretch = average XY distance from each alive player to the centroid
range_x = largest x − smallest x
pairwise distance = average XY distance across unique player pairs
hull area = area inside the outer polygon of alive-player XY positions
```

The convex hull is a geometric envelope, not literal controlled territory. It can cross walls or include areas no player can see. Euclidean distance also is not navigable route distance, firing-line visibility, or rotation time. These are inexpensive, reproducible descriptors whose tactical relevance must be tested rather than assumed.

A useful debugging surprise was zero spread or area. A single surviving player has zero pairwise spread, and fewer than three non-collinear players cannot form a positive-area polygon. The audited zero-spread observations had only one T alive. The arithmetic was valid; the interpretation “a tightly coordinated team” would have been wrong. Geometry needed survival context.

The geometry ablation reduced log loss from 1.0240 for the class prior to 0.9161 for logistic A1. This establishes predictive value in the development comparison. It does not show that every individual geometry column is independently useful or that the representation is optimal.

## 08 · Motion, combat, economy, and defense

Motion adds four inputs: average T speed over the one-second window, T centroid x/y velocity over that window, and bomb speed. Movement direction can distinguish similarly located groups that are approaching or leaving an area. The builder matches currently alive T players to their previous positions by identity and checks that the required prior states are present. Using the same player identities at both times prevents a casualty alone from appearing as centroid movement.

The motion addition was a weak overall positive in the linear model: log loss moved from 0.9161 to 0.9119. That small aggregate difference and horizon variation support cautious interpretation. V0 does not establish that a long sequence model would add value merely because a short movement summary helped a little.

Combat adds T and CT alive counts plus their difference, health sums plus their difference, and armor sums plus their difference. These nine variables explain whether a spatial pattern belongs to five healthy attackers or one damaged survivor. They also provide context for whether an attempted site approach is likely to survive long enough to produce a plant.

Combat was the strongest incremental linear addition: log loss improved from 0.9119 to 0.8359, with improvement at every evaluated horizon. CT health and armor are already used here. Full-observer requirements therefore begin before the explicit defense family is added.

Economy tests the sum of current equipment values for each side and their difference. It is a reasonable hypothesis that remaining equipment captures combat capability. It is not the same as a complete buy-round model, original purchase value, or future spending power. Adding these columns worsened logistic log loss to 0.8509 and XGBoost log loss from 0.8157 to 0.8274. The selected V0 excludes them.

Defense adds CT centroid x/y/z, stretch, x/y ranges, pairwise distance, hull area, T–CT centroid distance, minimum opponent distance, and mean nearest-opponent distance. These extend the question from “where are the attackers?” to “how do the two formations relate?” They still do not encode exact sight lines, utility effects, or tactical semantics.

Defense produced a particularly informative disagreement. Logistic A5 improved classification metrics relative to A3 but worsened log loss and Brier. XGBoost A5 improved the probability metrics. This motivated the interpretation that some defensive information may require nonlinear interactions, while leaving the precise causal mechanism unproven.

## 09 · Why sanity checks precede predictive evaluation

Feature validation had two layers. Numerical checks looked for missing or invalid values, negative quantities that should be non-negative, and implausible scales. Tactical checks followed observations through an actual round to see whether changing features matched the visible development of formations, movement, survival, and bomb state.

In the original 49-observation feature audit, geometry expanded during the middle of the round and contracted as attackers converged toward the eventual A plant. Movement summaries reflected periods of faster movement and slower positioning. Combat state explained the apparently suspicious zero-spread cases. T–CT centroid distance decreased as formations converged.

Those are construction checks. They make it more plausible that the feature pipeline represents what its names claim, but they cannot establish generalizable prediction. A perfectly implemented equipment feature can still fail ablation. Conversely, a model might appear strong because an incorrectly constructed feature leaks an outcome. Both correctness and predictive evaluation are needed.

The ordering matters: investigate suspicious features before celebrating a score. Otherwise an improvement can become a reason to defend a broken representation. This project keeps “implemented correctly,” “tactically interpretable,” and “useful on held-out matches” as different evidence statements.

## 10 · Match-grouped evaluation and out-of-fold predictions

Randomly splitting observation rows is too permissive here. A 10-second observation from a round could enter training while that same round's 20-second observation enters testing. Splitting only by round avoids that particular overlap but still shares match-specific conditions between training and testing.

V0 groups observations by demo_filename and uses five StratifiedGroupKFold folds with random_state 42. Stratification seeks a reasonable class distribution while the grouping constraint keeps an entire demo in one fold. Each fit trains on the other folds and predicts only the held-out fold. The code asserts that train and test match groups do not overlap.

```text
For each frozen fold:
  fit preprocessing and model using training matches only
  predict the held-out matches
  store predictions with observation keys and fold identity

Combine held-out predictions → out-of-fold (OOF) evaluation
```

The scaler for logistic regression is inside the training pipeline. It learns means and scales from the training portion of each fold, preventing information from the held-out feature distribution from entering preprocessing. The probability columns are explicitly aligned to A_PLANT, B_PLANT, NO_PLANT so that a label-order mismatch cannot silently corrupt metrics.

Later model comparisons reuse the frozen A1 fold assignments. This makes changes paired: competing models see the same held-out matches and observations. A gain is less likely to be an artifact of an easier split. The class-prior baseline also estimates its probabilities from training labels, rather than borrowing the full dataset's label distribution.

OOF predictions are held out from the fit that produced them. They are nevertheless development evidence once repeated model and feature choices are made after inspecting them. Match separation does not make the same 20 matches an untouched final test, nor does it guarantee that teams, competitions, or playing styles never recur across folds.

## 11 · Metrics: why a correct class is not enough

V0 is a probability forecaster. Log loss and Brier score are therefore primary numerical metrics, with calibration inspection alongside them. Accuracy, macro F1, and class-specific recall explain classification behavior but do not replace probability assessment.

Log loss is the average negative natural logarithm of the probability assigned to the true class. It strongly penalizes confidently wrong forecasts. In an illustrative A-plant example, assigning A probability 0.8 gives loss −ln(0.8) ≈ 0.223; assigning A 0.2 gives ≈ 1.609. Smaller is better. See the [scikit-learn log-loss definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).

The project's multiclass Brier implementation averages the sum of squared errors across all three probability columns. It is not divided by the number of classes. For probabilities (0.55, 0.20, 0.25) and a true A label (1, 0, 0), the illustrative per-observation score is 0.2025 + 0.0400 + 0.0625 = 0.3050. The multiclass convention used here ranges from 0 to 2; lower is better.

```text
Log loss = average[−ln(probability assigned to the true class)]
Brier = average[sum over classes (predicted probability − one-hot truth)²]
Accuracy = fraction whose largest-probability class is correct
```

Accuracy ignores how probability is distributed away from the largest class. Macro F1 computes an F1 score for each class and averages those scores equally, giving the smaller B class a visible role. Recall for A asks what fraction of true A observations the classifier identifies as A; it is different from the probability that an A prediction is correct.

Calibration asks whether stated confidence agrees with empirical frequency across groups of predictions. A model's 70% claims should succeed roughly 70% of the time in an appropriate evaluation population. Good ranking, high accuracy, and calibration are related but distinct properties. A relatively small overall calibration error can conceal a troublesome confidence range or horizon.

## 12 · The linear baseline and the ablation logic

A class prior asks whether any state information is necessary to beat knowledge of training-set outcome frequencies. Logistic regression then gives a regularized, comparatively simple model for testing the engineered representation. For each class it learns a score from weighted features; softmax converts the three scores into probabilities.

```text
score(class k) = intercept_k + sum_j weight[k,j] × standardized_feature[j]
P(class k) = exp(score_k) / sum_c exp(score_c)
```

The scores are linear in the selected inputs. The final probabilities are nonlinear through softmax, but this does not automatically provide arbitrary interactions between raw input features. For example, a bomb coordinate does not get a separately learned effect for every possible CT formation unless the representation or model supplies that interaction.

| Stage | Information set | Log loss ↓ | Brier ↓ | Accuracy ↑ | Macro F1 ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| A0 | Training class prior | 1.0240 | 0.6186 | 0.4913 | 0.2196 |
| A1 | Horizon + offensive geometry | 0.9161 | 0.5675 | 0.5142 | 0.4628 |
| A2 | A1 + motion | 0.9119 | 0.5612 | 0.5347 | 0.4823 |
| A3 | A2 + combat | 0.8359 | 0.4939 | 0.5994 | 0.5541 |
| A4 | A3 + economy | 0.8509 | 0.5011 | 0.5986 | 0.5495 |
| A5 | A3 + defense, no economy | 0.8408 | 0.4954 | 0.6151 | 0.5687 |

This is an information experiment as much as a model contest. Geometry earns a place, motion provides a small positive, and combat makes another substantial contribution. Economy is not retained simply because its hypothesis sounds sensible. Defense creates a question because classification improves while the primary probability metrics become slightly worse.

The model ladder originally included a geometry heuristic and later sequence models as possibilities. The frozen quantitative comparison here contains the class prior, logistic stages, and XGBoost variants. The website does not attach measured results to candidates that were only proposed.

## 13 · Why XGBoost was the next experiment

The mixed defense result suggested a concrete follow-up: hold the task, observations, folds, and feature families fixed, but use a model that can express feature interactions. The data is tabular and relatively small. A boosted-tree baseline tests that hypothesis before introducing a much larger sequence-modeling problem.

A decision tree partitions inputs through threshold questions. An illustrative branch might first ask about bomb position, then about opposing formation distance, then about surviving T count. The effect of one variable can depend on which previous questions were answered. This is how a tree can represent conditional structure. Those example questions are not an extracted explanation of the fitted V0 trees.

One shallow tree offers limited flexibility. Boosting builds an additive ensemble: each new stage changes the current prediction scores in a direction intended to reduce the training objective. It is more precise to describe this as fitting loss derivatives than to say that each tree merely memorizes previously misclassified rows. XGBoost also uses regularization to constrain tree complexity and leaf weights.

V0 uses the multiclass soft-probability objective so the model returns all three class probabilities. The fixed configuration has 300 boosting rounds and maximum depth 3. In the usual one-tree-per-class multiclass strategy, 300 rounds should not be described as only 300 total individual trees; each round can add a tree for each class.

The reason to use XGBoost is therefore testable: can a conservative nonlinear tabular model extract useful structure that the linear baseline did not? The claim is not that a more sophisticated algorithm must win. Its place is earned only by the controlled comparison.

## 14 · The exact XGBoost configuration and its tradeoffs

The local evaluation script fixes the configuration rather than running a large hyperparameter search over the same OOF results.

| Setting | V0 value | Role in this experiment |
| --- | ---: | --- |
| n_estimators | 300 | Number of boosting rounds |
| max_depth | 3 | Restrict the depth and complexity of individual trees |
| learning_rate | 0.03 | Shrink each boosting update |
| min_child_weight | 5 | Require sufficient Hessian weight before a child split; not a literal five-player or five-row threshold |
| subsample | 0.8 | Sample training observations for tree construction |
| colsample_bytree | 0.8 | Sample feature columns per tree |
| reg_alpha | 0.5 | L1 penalty on leaf weights |
| reg_lambda | 5.0 | L2 penalty on leaf weights |
| objective / num_class | multi:softprob / 3 | Predict three-class probability vectors |
| eval_metric | mlogloss | Multiclass log-loss evaluation setting |
| tree_method | hist | Histogram-based split construction |
| random_state | 42 | Fixed pseudorandom seed |
| n_jobs | −1 | Use available CPU threads |

Depth, shrinkage, sampling, and penalties all influence flexibility, but these choices are not proof of optimal regularization. A fixed seed aids reproducibility within the recorded environment; it does not remove statistical uncertainty or promise bit-for-bit identity across every future library and platform.

The parameter meanings are documented in the [official XGBoost parameter reference](https://xgboost.readthedocs.io/en/latest/parameter.html). The exact values above come from this project's evaluation script, not from a general recommendation in that reference.

The lack of a large search matters because there are only 20 matches. Repeatedly inspecting OOF scores, changing parameters, and selecting the best result can gradually tune to the evaluation set. A conservative initial configuration limits one source of that pressure. Feature and model selection still make this development evaluation, so the result requires fresh confirmation.

## 15 · What the nonlinear comparison actually found

| Model | Inputs | Log loss ↓ | Brier ↓ | Accuracy ↑ | Macro F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic A3 | 26 | 0.8359 | 0.4939 | 0.5994 | 0.5541 |
| XGB-A3 | 26 | 0.8157 | 0.4880 | 0.6167 | 0.5876 |
| XGB-A4 | 29 | 0.8274 | 0.4958 | 0.6080 | 0.5786 |
| XGB-A5 | 37 | 0.8024 | 0.4785 | 0.6285 | 0.5996 |

The A3-to-A3 comparison isolates the change in model family while keeping the information set fixed. XGBoost improves log loss from 0.8359 to 0.8157. The XGB-A3-to-XGB-A5 comparison then isolates adding defense within the nonlinear model, improving log loss to 0.8024. Defense improved XGBoost at every tested horizon in the frozen analysis.

XGB-A4 worsens probability quality relative to XGB-A3, so the economy hypothesis did not recover under this nonlinear baseline. XGB-A5 branches from A3 and excludes economy. Describing A5 as “all available features” would misrepresent the experiment.

The frozen selection is XGB-A5: the best development baseline among these comparisons, with offensive geometry, motion, combat, defense, and horizon. This supports the hypothesis that defensive information contains useful nonlinear or interaction-dependent structure. It does not identify a causal defensive mechanism, prove every defense column is necessary, or establish superiority on all unseen populations.

## 16 · Calibration: useful confidence with a visible weak spot

Top-label calibration compares the model's maximum probability with whether its chosen class was correct. The analysis uses confidence bins and a count-weighted expected calibration error (ECE). XGB-A5's overall top-label ECE is 0.0454, with classwise ECE of 0.0565 for A, 0.0354 for B, and 0.0432 for no plant.

The average hides a clear weakness: in the 0.6–0.7 confidence bin, 225 observations had mean confidence about 0.646 but observed accuracy about 0.524. That is an approximately 12.2 percentage-point overconfidence gap. The 30-second horizon also has the largest top-label ECE, about 0.0853.

| Minimum model confidence | Accuracy among selected predictions |
| --- | ---: |
| At least 0.50 | 66.8% |
| At least 0.60 | 72.7% |
| At least 0.70 | 81.2% |
| At least 0.80 | 88.6% |
| At least 0.90 | 94.9% |

These are nested, selected subsets. “94.9% accurate at confidence ≥0.90” describes 157 high-confidence observations, not the entire 1,268-observation dataset. Raising a threshold changes coverage, and the retained situations may be systematically easier. A deployable abstention or alerting policy would need explicit coverage, operating costs, and fresh-data validation.

No post-hoc calibration method is demonstrated by these numbers. This is an assessment of the baseline's probability behavior. ECE depends on binning and sample composition and is not a universal probability guarantee. The evidence supports inspecting confidence, not presenting all displayed probabilities as deployment-calibrated truth.

## 17 · Error structure: the main failure is not an A/B swap

The confusion matrix records true labels in rows and predicted labels in columns. It contains observations, not unique rounds or matches.

| True outcome → predicted outcome | A_PLANT | B_PLANT | NO_PLANT |
| --- | ---: | ---: | ---: |
| A_PLANT | 241 | 32 | 149 |
| B_PLANT | 35 | 107 | 81 |
| NO_PLANT | 130 | 44 | 449 |

The diagonal contains 797 correct predictions; the off-diagonal contains 471 errors. Direct site swaps are 32 + 35 = 67. Plant/no-plant boundary errors are 149 + 81 + 130 + 44 = 404. Therefore, 404 / 471 = 85.8% of classification errors cross the plant-occurrence boundary, while 14.2% directly confuse A with B.

Per-class recall is 57.11% for A, 47.98% for B, and 72.07% for no plant. The lower B recall is important even though B is the smallest class. Overall accuracy by itself would obscure that weakness.

This error grouping changes the next research question. It suggests that predicting whether a plant will happen is a larger difficulty than choosing the site once a plant is known to occur. That is an observed structure in V0's failures, not a proof that a particular replacement architecture must improve them.

## 18 · Decomposing the task without training a new model

The hierarchical analysis reuses the existing three-class probabilities. It first combines A and B to obtain plant probability, then renormalizes A and B for conditional site choice.

```text
P(plant) = P(A) + P(B)
P(A | plant) = P(A) / [P(A) + P(B)]
P(B | plant) = P(B) / [P(A) + P(B)]
```

The plant-versus-no-plant diagnostic has log loss 0.5740, binary Brier 0.1978, accuracy 0.6909, and F1 0.7066. The A/B diagnostic is evaluated only on the 645 observations whose true label is a plant; its accuracy is 0.7845 and log loss 0.4490. This conditioning uses true outcomes for analysis, not information available to an online predictor.

The 78.45% conditional site accuracy is not the original model's overall accuracy. It excludes all true no-plant observations and asks a different question. Likewise, binary Brier values use a different scoring expression from the three-class summed Brier above and should not be compared as if they measured the same task on the same scale.

Conditional site accuracy is 76.5%, 77.4%, 80.5%, and 83.0% across 10, 20, 30, and 40 seconds. The shrinking eligible populations still apply. These values do not establish a within-round causal improvement from waiting.

This decomposition is diagnostic. No separately trained hierarchical model is being evaluated here. It exposes where the present three-class model's information is useful and where its task may benefit from a more focused treatment.

## 19 · What permutation importance does and does not explain

The group permutation experiment fits the same fold models, then disrupts one feature family in held-out observations while keeping the fitted models fixed. All columns within that family receive the same row permutation. This preserves relationships inside the shuffled family while breaking its alignment with the original outcome and other feature groups.

The experiment repeats 20 times, permuting within each held-out fold. The reported value is the change in pooled OOF loss relative to the unpermuted baseline. Positive change means destroying that information made the model worse.

| Shuffled family | Change in three-class log loss |
| --- | ---: |
| Offensive geometry | +0.2806 |
| Combat | +0.2315 |
| Defense | +0.0274 |
| Motion | +0.0112 |
| Horizon | −0.0002 |

These numbers are not percentages, do not sum to 100%, and should not be added into a total contribution. Correlation and redundancy mean that one family can substitute for another. Shuffling can also create combinations unlike real game states, so the result describes sensitivity of this fitted predictor to that perturbation, not a causal effect of moving players or changing health.

Task-specific importance sharpens the interpretation. For plant versus no plant, shuffling combat increases log loss by 0.2287, compared with 0.0287 for offensive geometry. For A versus B given a true plant, shuffling offensive geometry increases log loss by 0.4953, compared with 0.0056 for combat. This supports the separation between plant feasibility and site choice.

The near-zero horizon permutation result does not prove that time is irrelevant to Counter-Strike. Other state features can encode progression indirectly, and the explicit horizon may add little once those are known in this model and dataset. Retaining a frozen contract and testing a targeted simplification on fresh evidence are separate decisions.

## 20 · Robustness, uncertainty, and the decision to freeze

An aggregate score can improve because of a few unusually favorable matches. The robustness analysis therefore compares XGB-A5 with Logistic-A3 separately for each held-out match. XGB-A5 wins on log loss for 13 of 20 matches and on Brier for 13 of 20. Seven matches still regress on each metric.

The equal-match mean log-loss delta is −0.0401 and the median is −0.0285, where delta means XGB-A5 minus Logistic-A3. The negative median shows that the improvement is not explained solely by one exceptionally favorable match. It does not remove the practical significance of the regressions.

There are two different averages here. The headline pooled observation-level difference is approximately 0.8024 − 0.8359 = −0.0336. The robustness mean is −0.0401 because it averages match deltas equally instead of giving matches with more eligible observations greater weight. Both are valid summaries of different estimands.

A paired match-level bootstrap resamples the 20 match comparisons with replacement 100,000 times. Each resample preserves the model comparison within a match and averages the selected match deltas. The 2.5th and 97.5th percentiles provide the reported interval.

| Metric | Mean match delta | 95% bootstrap interval | Resamples with delta below zero |
| --- | ---: | --- | ---: |
| Log loss | −0.0401 | [−0.0844, −0.0034] | 0.9855 |
| Brier | −0.0160 | [−0.0355, +0.0024] | 0.9550 |

The log-loss interval remains below zero, so its directional evidence is stronger in this analysis. The Brier interval crosses zero. The bootstrap fractions are empirical resampling summaries, not posterior probabilities that the model is universally better and not final confirmatory significance tests after selection.

Twenty matches remain a small development sample, potentially with recurring teams and shared competitive context. Bootstrapping that sample does not create genuinely new matches, eliminate selection effects, or solve distribution shift. V0 freezes the evaluation so future work cannot repeatedly optimize these results and later present them as an unbiased final test.

## 21 · Limitations and the honest completion boundary

V0's offline feature construction, baseline comparisons, and development analyses have recorded results. That is the completion claim. The broader tactical intelligence system is not complete.

The main boundaries are:

- One map and a limited match population. Generalization to other maps, levels of play, teams, or future game changes is not established.
- Correlated observations. The dataset has 20 match groups, not 1,268 independent tactical experiments.
- Full observer inputs. CT location, health, armor, and survival features are not all normal player-view information. A partial-observation model needs a different contract and evaluation.
- Outcome rather than intention. The labels identify recorded plant outcomes, not tactical plans, round winners, or the quality of a chosen strategy.
- Simplified representation. Euclidean formations do not capture sight lines, map navigation, utility semantics, communication, or every relevant tactical interaction.
- Development selection. Feature and model choices used the same OOF evidence. There is no untouched final-test result in the frozen summary.
- Imperfect calibration and class performance. Mid-confidence overconfidence, 30-second calibration weakness, and lower B recall remain visible.
- Serving work remains. No demonstrated online latency, replay parity, observer-feed parity, production monitoring, or deployable model service follows from the offline scores.

The website documents these findings. Its deployment is a publication of the research record, not a deployment of the prediction model. There is no simulated live predictor masquerading as an operational system.

## 22 · The V1 hypothesis and what would count as evidence

V0's errors and importance analysis motivate a candidate two-stage architecture: one model estimates plant feasibility; a second estimates site choice conditional on a plant. A possible probability composition would be:

```text
q = predicted P(plant)
r = predicted P(A | plant)

P(A) = q × r
P(B) = q × (1 − r)
P(NO_PLANT) = 1 − q
```

This is a V1 hypothesis, not an implemented improvement. The diagnostic decomposition of V0 does not prove that separately training two models will perform better. Separate stages can introduce their own estimation errors, and selecting a conditional training population requires care. The original three-class XGB-A5 remains the comparison baseline.

A useful next experiment would freeze the new hypothesis and evaluation plan, obtain untouched matches or use an appropriate nested design, compare probability metrics under the same observation contract, inspect calibration and per-class outcomes, and revisit match-level robustness. The proposed architecture should improve the actual three-class forecasting task, not only a conditional metric that excludes hard examples.

Replay and serving parity are another clear engineering direction. Historical states should pass through the same feature and inference path intended for online use, with explicit timing, missingness, and latency checks. These are related to the eventual product, but they should not be confused with evidence that the hierarchical model is better.

Longer-term ideas such as opponent modeling, historical retrieval, tactical concepts, transcript-assisted labeling, utility knowledge, and counter-strategy evaluation remain distinct research tracks. V0's contribution is a defensible starting point and a disciplined way to decide what deserves investigation next.

## 23 · Source map and a practical reading path

For the fastest understanding, read the task and time boundary first, then the ablation tables, error structure, and limitations. For the design story, follow the chapters in order: every model change makes more sense after the source investigations and feature checks that motivated it.

The local research record provides several evidence levels:

- `docs/v0_evaluation_summary.md`: frozen task, result interpretation, calibration, robustness, uncertainty, and the V1 hypothesis.
- `docs/v0_feature_candidate_matrix.md`: candidate representations and their original rationale. A candidate or “ready for ablation” label is not a final keep decision.
- `docs/v0_feature_evidence.md`: the initial 49-observation construction and tactical checks, which precede predictive evaluation.
- `src/cs2_tactical_intelligence/v0/features.py`: consolidated feature construction used by the batch dataset builder.
- `scripts/build_v0_dataset.py`, `scripts/evaluate_v0_a1.py`, `scripts/evaluate_v0_stage.py`, and `scripts/evaluate_v0_xgboost.py`: dataset orchestration, grouped evaluation, feature contracts, and fixed model settings.
- `scripts/analyze_v0_*.py`: calibration, confusion, hierarchical diagnostics, permutation sensitivity, match robustness, and bootstrap analysis.
- Frozen CSV tables on the [results and evidence page](../site/pages/evidence.html): the numerical record used in this explanation.

The original CSGO2 conversation supplies the project history and request to explain not just XGBoost but the design decisions and investigations that led to it. The checked-in code and frozen artifacts supply the quantitative and implementation details. Illustrations in this chapter explain concepts; they are not additional measured experiments.
