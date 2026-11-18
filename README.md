# ML_HACKATHON_TEAM8

## PathSnap

PathSnap generates GPS-to-road candidate features and ground-truth labels for map
matching. The candidate pipeline uses a 100-metre search radius, retains 30
candidates per observation, and aligns labels using the existing 150-metre
monotone route-alignment search.

### Member 1: preprocessing and feature generation

Generate the training features and observation summary from the repository root:

```powershell
cd C:\Users\karth\ML_HACKATHON_TEAM8\pathsnap
python scripts/generate_features.py --root "C:\Users\karth\ML_HACKATHON_TEAM8\Student folder\train" --out outputs
```

The script writes Parquet and compressed CSV datasets under `pathsnap/outputs/`.
It also writes `generation_summary.md` and the existing pipeline compatibility
files `candidates.csv.gz` and `observations.csv.gz`.

- [Label definition](pathsnap/docs/label_definition.md)
- [Generated training summary](pathsnap/outputs/generation_summary.md)

#### Generated training data

The latest executed training generation processed 80 trips and produced:

| Metric | Value |
|---|---:|
| Observations | 208,417 |
| Candidate rows | 3,247,671 |
| Valid labeled observations | 199,233 |
| Positive labels | 186,931 |
| Negative labels | 2,996,015 |
| Label coverage | 95.5934% |
| Candidate recall | 93.8253% |
| Candidate miss rate | 6.1747% |
| Recall@1 | 51.6566% |
| Recall@3 | 88.4934% |
| Recall@5 | 91.3579% |
| Recall@10 | 92.9454% |

Trips 54, 33 and 2 have candidate recalls of 21.5812%, 46.2356% and 96.2754%,
respectively. Trip 54 is primarily affected by radius misses and a group of
observations without candidates; trip 33 is also primarily affected by radius
misses. The candidate radius and 30-candidate cap were preserved.

No processing failures occurred in the latest training generation.

### Tests

```powershell
cd C:\Users\karth\ML_HACKATHON_TEAM8\pathsnap
python -m pytest -q
```

The existing suite contains 24 tests. Generated labels, feature leakage and
trip-level train/validation isolation are also checked during pipeline validation.

## Member 2: model training and optimisation

See [setup and reproduction instructions](pathsnap/MEMBER2_README.md) for the
five-model comparison, trip-grouped validation, Bayesian tuning and candidate scoring.

- [Experiment report and plots](member2_results/MEMBER2_REPORT.md)
- [Testing summary](member2_results/validation/TESTING_SUMMARY.md)
- [Published results guide](member2_results/README.md)

Raw datasets, prepared feature tables and trained model binaries are excluded from
Git. The training scripts reproduce the artifacts locally from the supplied dataset.
