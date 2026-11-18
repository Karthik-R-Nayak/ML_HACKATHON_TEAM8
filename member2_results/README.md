# Member 2 experiment results

This snapshot contains measured results from the completed local experiments:
five model families, three-fold trip-grouped CV on 64 development trips, a frozen
16-trip holdout, 12 Bayesian TPE trials for histogram boosting, and student-testing
comparisons after refitting on all 80 training trips.

The selected artifact is **untuned Random Forest**, selected using CV trip-mean
observation Top-1: 83.40% CV, 85.32% holdout and 87.81% student testing.
Untuned histogram boosting reached 87.94% on student testing; those student scores
did not change the frozen CV selection. Tuning did not show a consistent held-out gain.

KNN alone uses up to 100 sampled training observations per trip; every candidate of
each selected observation is retained. All models use complete validation/testing
partitions. Other models use all valid training rows. The report documents this limitation.

- [Report and figures](MEMBER2_REPORT.md)
- [Full validation summary](validation/TESTING_SUMMARY.md): 37 pytest tests, 13 real-data/artifact checks and nine CLI checks passed.
- [Baseline CV metrics](baseline_cv.csv), [student metrics](student_comparison.csv), [holdout metrics](holdout_comparison.csv)
- [Bayesian trial log](tuning_trials.csv) and [frozen selection](selection.json)
- `cv/` and `tuning/`: fold metrics and per-trip results.
- `data_metadata/`: preparation manifests, audits and feature-missingness counts.

The source commit and hashes in the provenance files describe the code used when
the experiments ran. Member 1's core pipeline has not been modified by Member 2.

Raw data, full generated tables, prediction arrays, SQLite state and trained model
binaries are intentionally excluded from this compact Git snapshot. Follow
[the reproduction instructions](../pathsnap/MEMBER2_README.md) to generate them.
Model checksums refer to local generated artifacts; they are not download links.
The scoring interface is `pathsnap.ml.predict.score_candidates` and requires no labels.

HMM/Viterbi, route-continuity evaluation, the API and frontend remain separate work.
