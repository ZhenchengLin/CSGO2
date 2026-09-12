"""Native MathML for the guide's fixed, documented equations (no runtime CDN)."""
from html import escape

def tag(name, body, **attrs):
    attributes=''.join(f' {key}="{escape(str(value), quote=True)}"' for key,value in attrs.items())
    return f'<{name}{attributes}>{body}</{name}>'
def mi(s): return tag('mi', escape(str(s)))
def mn(s): return tag('mn', escape(str(s)))
def mo(s): return tag('mo', escape(s))
def tx(s): return tag('mtext', escape(s))
def row(*s): return tag('mrow',''.join(s))
def sub(a,b): return tag('msub',a+b)
def sup(a,b): return tag('msup',a+b)
def frac(a,b): return tag('mfrac',a+b)
def par(a): return row(mo('('),a,mo(')'))
def summ(lo,hi,body): return row(tag('munderover',mo('∑')+lo+hi),body)
def prob(a): return row(mi('P'),par(a))
def norm(a): return row(mo('‖'),a,mo('‖'))
def eq(a,b): return row(a,mo('='),b)
def math(content,display=False):
    return '<math xmlns="http://www.w3.org/1998/Math/MathML"'+(' display="block"' if display else '')+'>'+content+'</math>'
A=tx('A');B=tx('B');NO=tx('NO');plant=tx('plant')
pa=prob(A);pb=prob(B);pn=prob(NO);pp=prob(plant)
pab=row(pa,mo('+'),pb)
c=mi('c');m=mi('m');n=mi('n');i=mi('i');j=mi('j');k=mi('k')
r_i=sub(mi('r'),i);r_xy=sub(mi('r'),row(i,mo(','),tx('xy')));c_xy=sub(c,tx('xy'))
unit_n=frac(mn(1),n)
z_k=sub(mi('z'),k);p_ik=sub(mi('p'),row(i,mo(','),k));y_ik=sub(mi('y'),row(i,mo(','),k))

# Each key corresponds to an explicit math fence in the source guide.
EQUATIONS={
 'forecast': ('The prediction target',[
 ('Conditional outcome',prob(row(mi('Y'),mo('|'),sub(mi('S'),row(mn(0),mo(':'),mi('t')))))),
 ('Three possible outcomes',row(mi('Y'),mo('∈'),mo('{'),tx('A_PLANT'),mo(','),tx('B_PLANT'),mo(','),tx('NO_PLANT'),mo('}')))
 ], 'S₀:ₜ is the state history available through time t. V0 summarizes this history as an engineered feature vector.'),
 'timing': ('The observation boundary',[
 ('Requested tick',eq(sub(mi('τ'),tx('request')),row(sub(mi('τ'),tx('freeze')),mo('+'),tx('round'),par(row(mi('h'),mo('·'),mn(64)))))),
 ('Resolved snapshot',row(sub(mi('τ'),tx('request')),mo('≤'),sub(mi('τ'),tx('resolved')),mo('≤'),sub(mi('τ'),tx('request')),mo('+'),mn(1))),
 ('Before the round ends',row(sub(mi('τ'),tx('obs')),mo('<'),sub(mi('τ'),tx('end')))),
 ('Before any plant',row(tx('no plant event'),mo('∨'),sub(mi('τ'),tx('obs')),mo('<'),sub(mi('τ'),tx('plant'))))
 ], 'h = observation horizon in seconds; the measured demo clock is 64 ticks per second. The resolver uses the first snapshot at or after the request and observed lateness is at most one tick. Both eligibility conditions must hold.'),
 'geometry': ('Formation geometry',[
 ('Team centroid',eq(c,row(frac(mn(1),m),summ(eq(i,mn(1)),m,r_i)))),
 ('Mean stretch',eq(sub(mi('s'),tx('xy')),row(frac(mn(1),m),summ(eq(i,mn(1)),m,norm(row(r_xy,mo('−'),c_xy)))))),
 ('Horizontal range',eq(sub(mi('R'),mi('x')),row(sub(tx('max'),i),sub(mi('x'),i),mo('−'),sub(tx('min'),i),sub(mi('x'),i)))),
 ('Mean pairwise distance',eq(sub(mi('d'),tx('pair')),row(frac(mn(2),row(m,par(row(m,mo('−'),mn(1))))),tag('munder',mo('∑')+row(i,mo('<'),j)),norm(row(r_xy,mo('−'),sub(mi('r'),row(j,mo(','),tx('xy')))))))),
 ('Convex hull area',eq(sub(mi('A'),tx('hull')),row(tx('area'),par(row(tx('conv'),par(row(mo('{'),r_xy,mo('}'))))))))
 ], 'm = number of alive players; rᵢ = player position; xy = horizontal plane. Pairwise distance is defined as zero for fewer than two players; hull area is zero for fewer than three non-collinear players.'),
 'metrics': ('How forecasts are scored',[
 ('Log loss',eq(sub(mi('L'),tx('log')),row(mo('−'),unit_n,summ(eq(i,mn(1)),n,row(tx('ln'),par(sub(mi('p'),row(i,mo(','),sub(mi('y'),i))))))))),
 ('Multiclass Brier',eq(sub(mi('L'),tx('Brier')),row(unit_n,summ(eq(i,mn(1)),n,summ(eq(k,mn(1)),mn(3),sup(par(row(p_ik,mo('−'),y_ik)),mn(2))))))),
 ('Accuracy',eq(tx('Accuracy'),row(unit_n,summ(eq(i,mn(1)),n,row(mi('𝟙'),mo('['),sub(tag('mover',mi('y')+mo('^')),i),mo('='),sub(mi('y'),i),mo(']'))))))
 ], 'n = observations; pᵢₖ = predicted probability for class k; yᵢₖ = one-hot truth; ŷᵢ = predicted class. Lower log loss and Brier are better; higher accuracy is better.'),
 'logistic': ('From features to probabilities',[
 ('Linear class score',eq(z_k,row(sub(mi('b'),k),mo('+'),summ(eq(j,mn(1)),mi('d'),row(sub(mi('w'),row(k,mo(','),j)),sub(tag('mover',mi('x')+mo('~')),j)))))),
 ('Softmax probability',eq(prob(eq(mi('Y'),k)),frac(sup(mi('e'),z_k),summ(eq(c,mn(1)),mn(3),sup(mi('e'),sub(mi('z'),c))))))
 ], 'd = input count; x̃ⱼ = standardized feature; wₖⱼ = learned weight; bₖ = intercept. Softmax makes the three class probabilities sum to one.'),
 'conditional': ('Separate occurrence from site choice',[
 ('Plant occurrence',eq(pp,pab)),
 ('A, given a plant',eq(prob(row(A,mo('|'),plant)),frac(pa,pab))),
 ('B, given a plant',eq(prob(row(B,mo('|'),plant)),frac(pb,pab)))
 ], 'The conditional probabilities renormalize A and B by their combined probability. These diagnostics reuse the V0 outputs; they do not train a new model.'),
 'hierarchy': ('A possible V1 probability model',[
 ('Stage 1 · feasibility',eq(mi('q'),pp)),
 ('Stage 2 · site choice',eq(mi('r'),prob(row(A,mo('|'),plant)))),
 ('A plant',eq(pa,row(mi('q'),mo('·'),mi('r')))),
 ('B plant',eq(pb,row(mi('q'),par(row(mn(1),mo('−'),mi('r')))))),
 ('No plant',eq(prob(tx('NO_PLANT')),row(mn(1),mo('−'),mi('q'))))
 ], 'q estimates plant probability; r estimates A conditional on a plant. The composed outputs sum to one. This architecture is a hypothesis, not a demonstrated improvement.')
}

def render_equations(key):
    title,expressions,note=EQUATIONS[key]
    return '<figure class="equation-panel"><figcaption>'+escape(title)+'</figcaption>'+''.join('<div class="equation-row"><span class="equation-label">'+escape(label)+'</span><div class="equation-scroll" tabindex="0" role="region" aria-label="'+escape(label)+' equation">'+math(expression,True)+'</div></div>' for label,expression in expressions)+'<p class="equation-note">'+escape(note)+'</p></figure>'

INLINE={
 '32 + 35 = 67':eq(row(mn(32),mo('+'),mn(35)),mn(67)),
 '149 + 81 + 130 + 44 = 404':eq(row(mn(149),mo('+'),mn(81),mo('+'),mn(130),mo('+'),mn(44)),mn(404)),
 'P(A) = 0.55':eq(pa,mn('0.55')),
 'P(B) = 0.20':eq(pb,mn('0.20')),
 'P(NO) = 0.25':eq(pn,mn('0.25')),
 '−ln(0.8) ≈ 0.223':row(mo('−'),tx('ln'),par(mn('0.8')),mo('≈'),mn('0.223')),
 '−ln(0.2) ≈ 1.609':row(mo('−'),tx('ln'),par(mn('0.2')),mo('≈'),mn('1.609')),
 '0.2025 + 0.0400 + 0.0625 = 0.3050':eq(row(mn('0.2025'),mo('+'),mn('0.0400'),mo('+'),mn('0.0625')),mn('0.3050')),
 '404 / 471 = 85.8%':eq(frac(mn(404),mn(471)),row(mn('85.8'),mo('%'))),
 '0.8024 − 0.8359 = −0.0336':eq(row(mn('0.8024'),mo('−'),mn('0.8359')),row(mo('−'),mn('0.0336'))),
 '45 + 63 = 108':eq(row(mn(45),mo('+'),mn(63)),mn(108)),
 '225 + 113 + 181 + 56 = 575':eq(row(mn(225),mo('+'),mn(113),mo('+'),mn(181),mo('+'),mn(56)),mn(575)),
 '575 / 683 = 84.2%':eq(frac(mn(575),mn(683)),row(mn('84.2'),mo('%'))),
 '0.838182 − 0.887596 = −0.049414':eq(row(mn('0.838182'),mo('−'),mn('0.887596')),row(mo('−'),mn('0.049414'))),
}

def format_inline_equations(value):
    for source,expression in INLINE.items():
        value=value.replace(escape(source,quote=False), '<span class="inline-equation">'+math(expression)+'</span>')
    return value
