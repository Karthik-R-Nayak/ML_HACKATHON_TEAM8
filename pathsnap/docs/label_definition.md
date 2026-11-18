# PathSnap Label Definition

## Candidate label

Each row is a candidate road arc for one GPS observation.

- **`label = 1`**: the candidate arc is the ground-truth arc selected by route-to-GPS alignment for that observation.
- **`label = 0`**: the observation is valid for supervised learning, but this candidate arc is not the aligned ground-truth arc.
- **`label_valid`**: `true` only when ground-truth alignment produced a feasible state for the observation. Invalid observations must be excluded from supervised training.

At most one candidate in a valid labeled observation has `label = 1`. A labeled observation may have no positive row if the aligned arc was not generated among its candidate arcs.

## Ground-truth route alignment

Alignment is performed only when generating labels. It uses the road network and the ground-truth route with a monotone Viterbi search:

1. Candidate route positions are limited to route arcs within 150 metres of the GPS point.
2. Route progress must remain monotone, with only the configured small backwards tolerance on the same arc.
3. The search penalizes mismatch between GPS displacement and route progress.
4. An observation is marked `aligned` when a consistent state exists; otherwise it is `no_route_arc_nearby` or `dropped_infeasible`.

The aligned route position and arc become `aligned_route_pos` and the ground-truth label target. Alignment output is never used as an inference feature.

## Unlabeled observations

Observations without route files or with failed alignment are not supervised training examples. Their candidate rows retain `label_valid = false`; `label` is not a positive or negative training target for these rows.

`training_matrix()` removes rows with `label_valid == true` before constructing `X`, `y`, and trip groups.

## Candidate recall

Candidate recall is calculated over observations with alignment status `aligned`:

\[
\text{candidate recall} =
\frac{\text{aligned observations whose ground-truth arc is among generated candidates}}
{\text{aligned observations}}
\]

The miss rate is `1 - candidate recall`. A radius miss means the aligned arc was outside the 100-metre candidate search radius. A cap miss means the aligned arc was within the radius but fell outside the 30 nearest retained candidates.

Recall@K measures the fraction of aligned observations for which the ground-truth candidate has rank no greater than K (`1`, `3`, `5`, or `10`). Missing candidates do not satisfy any K threshold.

## Leakage prevention

Ground-truth routes and labels are used only after candidate and base GPS/network features are generated. `FEATURE_COLUMNS` contains only GPS, candidate-distance/rank, heading, road-geometry, and candidate-count measurements.

Columns such as `label`, `label_valid`, `aligned_route_pos`, `gt_dist_m`, `gt_rank`, and alignment status must never enter `X`. This keeps offline training and real inference behavior independent of route information.
