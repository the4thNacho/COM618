# Bayes' Rule (Bayes' Theorem)

Purpose: Update the probability of a hypothesis given new evidence — commonly used for diagnostic testing, spam filtering, and decision-making under uncertainty.

## Formula

P(A | B) = (P(B | A) * P(A)) / P(B)

Where:

- P(A | B): probability of A given B (posterior)
- P(B | A): probability of B given A (likelihood)
- P(A): prior probability of A
- P(B): marginal probability of B

## How to use

1. Write down prior P(A) and likelihood P(B|A).
2. Compute P(B) = P(B|A)P(A) + P(B|¬A)P(¬A).
3. Apply the formula to get the posterior P(A|B).

## Example (medical test)

Prevalence P(Disease) = 0.01 (1%)

Sensitivity P(Positive|Disease) = 0.90

Specificity P(Negative|NoDisease) = 0.95 → P(Positive|NoDisease)=0.05

P(Disease|Positive) = (0.90 * 0.01) / (0.90*0.01 + 0.05*0.99)

= 0.009 / (0.009 + 0.0495) ≈ 0.1538 (≈ 15.4%)

Note: Even with a good test, low prevalence leads to many false positives.

## Tips

- Always check the base rate (prior) — avoid the base-rate fallacy.
- Use likelihood ratios for quick screening: LR+ = Sensitivity/(1−Specificity).

References: conditional probability, likelihood ratio, diagnostic testing.
