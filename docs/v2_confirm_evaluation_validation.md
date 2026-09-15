# V2 Independent Confirmation — Evaluation Validation

## Status

PASS

The frozen V0 and H2 models were evaluated once on the untouched
30-match D_CONFIRM dataset.

No retraining, retuning, threshold selection, or second confirmation
inference was performed.

## D_CONFIRM

- Matches: 30
- Observations: 2269
- Dataset SHA256:
  e781dbebe0956322476cd8841675ad9a22e9d680c6dec7a47f5d23c35829c367

## Formal confirmation results

### V0

- Log loss: 0.835090
- Brier score: 0.508136
- Accuracy: 0.589246
- Macro F1: 0.541890

### H2

- Log loss: 0.836643
- Brier score: 0.508533
- Accuracy: 0.589246
- Macro F1: 0.535487

### H2 minus V0

- Log loss: +0.001553

Lower log loss is better, therefore H2 did not improve the primary
confirmation metric over V0.

## Horizon log loss

- 10 s: V0 0.902952, H2 0.903492, delta +0.000540
- 20 s: V0 0.858702, H2 0.857766, delta -0.000936
- 30 s: V0 0.793310, H2 0.800057, delta +0.006747
- 40 s: V0 0.773230, H2 0.772956, delta -0.000274

The H2 effect was not consistently favorable across horizons.

## Equal-match paired comparison

- Matches: 30
- H2 better matches: 12
- Mean log-loss delta H2-V0: +0.003106
- 95% bootstrap CI: [-0.007585, +0.014124]

The confidence interval crosses zero.

## Probability-sum warning validation

scikit-learn emitted probability-sum warnings for V0.

Post-evaluation inspection used only the already-saved prediction
artifact. Models were not rerun.

### V0

- Minimum probability sum: 0.9999999199062586
- Maximum probability sum: 1.0000000819563866
- Maximum absolute deviation from 1: 8.1956386566162109e-08
- Rows with deviation > 1e-7: 0
- Rows with deviation > 1e-6: 0
- Raw log loss: 0.835090028834
- Normalized log loss: 0.835090029520
- Difference: -6.860751966542e-10

### H2

- Maximum absolute deviation from 1: 1.1102230246251565e-16
- Raw log loss: 0.836642972629
- Normalized log loss: 0.836642972629

Conclusion: the warning is a floating-point normalization artifact and
does not materially affect the reported confirmation metrics.

## Scientific decision

V0 remains the primary model.

H2 remains a documented secondary challenger, but its development-stage
Site calibration improvement did not translate into an improvement in
the primary three-class confirmation log-loss metric.

No H2 promotion is made from this experiment.
