# PathSnap testing summary

**Status: PASS for the implemented ML scope.**

| Check | Result |
| --- | --- |
| Full pytest suite | 37 passed; zero failures, errors or skips |
| Real data/artifact checks | 13 passed |
| CLI and cached resume checks | 9 passed |
| Saved model configurations | All six ran on all 366,633 student candidates |
| Probability/metric reproduction | All saved probabilities, F1, log loss, confusion matrices and Top-1 reproduced |
| Feature regeneration without reference routes | All 10 student trips matched prepared features |
| Data and split validation | 80 training + 10 student trips consistent; no trip leakage |
| Bayesian study | 12 completed trials; database, checkpoint and fold scores consistent |
| Malformed input | Missing features, duplicate candidate IDs and infinite features rejected |
| Empty input | Accepted by all six model interfaces |

No models were retrained and no optimisation trials were added. Original experiment metrics and selected model were preserved.

Two non-blocking warnings remain: Windows physical-core detection falls back to logical cores, and an existing Member 1 test triggers a pandas FutureWarning. Neither caused a failure.

The selected Random Forest still reproduces 87.806% trip-averaged Top-1 on the 10 student trips. These scores use route-aligned labels.

HMM/Viterbi, FastAPI, frontend integration and two-laptop deployment are outside the completed Member 2 scope and have not been verified here.

Evidence: [pytest XML](pytest-results.xml), [artifact checks](artifact-validation.json), [CLI checks](cli-validation.json).
