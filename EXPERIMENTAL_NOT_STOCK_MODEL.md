# EXPERIMENTAL MODEL CODE: NOT THE MULTI-AREA MODEL WHEN ITS NEW SETTINGS ARE USED

This tree is a separate, confined copy of our beNNch-models branch `save-cycle-log` (1f63330), made for one experiment:
the lookahead experiment in `/home/magnus/Projects/sp4-direction-scoping/lookahead_percolation_2026-10-28/`
(`NEST_DESIGN.md`). Branch `EXPERIMENTAL-NOT-STOCK-interarea-delay-floor`. It has no git remotes; the owner adds one
only to push this branch.

**Three new simulation settings,** at their defaults the model as on `save-cycle-log`:
- `interarea_delay_floor_steps` (default 0): inter-area delays are redrawn below the floor minus half a step. With a
  floor, **this is not the multi-area model.**
- `area_split` (default `none`): a file in `multi-area-model/multiarea_model/area_splits/` that puts each area on one
  process. All units are created in one call, and each area takes ids congruent to its process. Processes with fewer
  units keep idle units: unconnected, never recorded, with no input, and frozen, so NEST does not update them. With a split, **this is not the multi-area model's
  placement.**
- `remote_exchange_slices` (default 1): passed to NEST only when above 1, which needs the experimental NEST
  `EXPERIMENTAL-NOT-STOCK-windowed-remote-exchange`.

The settings come in as the optional arguments 12 to 14 of `run_benchmark_createParams.py`. Every run that uses them is
data of a changed model: it goes only to `nest/nest_cluster_data/CHANGED_NEST_NOT_STOCK/`, with a marker, never among
the reference runs.
