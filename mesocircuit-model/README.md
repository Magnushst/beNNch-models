# The mesocircuit as a ci-beNNch benchmark

The mesocircuit of Senk and others (Cerebral Cortex 2024), parameter set `mesocircuit_MAMV1`: area V1 of the
multi-area model, upscaled to a 4 × 4 mm² torus. It has 3,181,344 units in 8 cortical populations and a silent thalamic
one, and about 7.05 × 10⁹ connections by NEST's spatial `pairwise_bernoulli`, with distance-dependent delays. It is
built by its own code, INM-6/mesocircuit-model at commit `f59ada6b3c3efb1fa18f96c638dd1741ebfb77e5`
(GPL-3.0, `LICENSE.rst`).

## Files

| File | What it is |
|---|---|
| `mesocircuit/helpers/mpiops.py`, `mesocircuit/parameterization/{base_network_params,base_simulation_params,helpers_network}.py`, `mesocircuit/__init__.py`, `parametersets.py` (the model's `scripts/parametersets.py`), `LICENSE.rst` | The model's own files, unchanged (byte-equal to the commit's). |
| `mesocircuit/simulation/network.py` | The model's file as its pull request 239 changes it (heplesser/mesocircuit-model at 11cea50, closed unmerged), byte-equal to that version. It differs from f59ada6 in two places. The positions of a rank's units are read from `pop.spatial["positions"]` with the ids `pop[pop.local].global_id`, a work-around for NEST #3706, where slicing a collection loses its spatial metadata. `np.NINF` and `np.Inf` become `-np.inf` and `np.inf`, the same values. Neither change touches the network. |
| `run_benchmark.py` | Derives the parameters as the model's `mesocircuit_framework.py` does, builds the network with the model's `Network.create()` and `Network.connect()` as its `run_network.py` does, then runs the warm-up and the measured period with `nest.Prepare` and `nest.Run`, as the multi-area model's benchmark does. |

## What a run records

Each rank writes two files into `data/`, as the PD14 benchmark does, so ci-beNNch's `collect_timer_data.py` reads
them unchanged:
- `logfile_<rank>`: the kernel status with the warm-up subtracted from the simulation timers, the Python timers, NEST's
  memory (virtual size) and the rank's peak resident memory;
- `cycle_time_log_<rank>.npz`: NEST's whole per-cycle log.

The seed, the threads and the durations come from the ci-beNNch config: 500 ms of warm-up, then 1,000 ms measured. No
recording devices are used, as in the multi-area model's benchmark.

The model's `create()` also writes its node ids and every unit's position into `model_output/raw_data/`. `connect()`
ends with `nest.Prepare()` and `nest.Cleanup()`, so `py_time_connect` includes NEST's preparation of the connection
infrastructure.

NEST's minimum delay here is 0.5 ms, so a cycle is 5 steps. The 1,000 ms measured are 2,000 cycles.

## Checked before any run

- The copied files are byte-equal to the commit's, and `network.py` to pull request 239's.
- The parameter derivation in `run_benchmark.py` gives a `net_dict` equal, value by value, to the one the engine's
  specification was made from. This was checked with the model's own code in an isolated environment (numpy 2.1.3),
  without NEST.
- Every NEST call `network.py` makes exists in NEST 3.10. That alone was not enough: the first run (pipeline 539577,
  2 Nov) failed in `create()`, where f59ada6 calls `GetPosition` on `GetLocalNodeCollection(pop)`, which raises
  `LayerExpected` in this NEST (#3706). In this NEST, `FreeLayer::get_status` returns a rank's own positions in the
  rank-local order of the collection (`nestkernel/free_layer.h`), the order of `pop[pop.local]`. So pull request 239's
  length assertion holds here.
- The Hambach stack has numpy 1.25.2, h5py 3.12.1 and mpi4py 4.0.1 (pip list of the multi-area model's Hambach run of
  31 Oct).
- `run_benchmark.py` parses once its four placeholders are filled. No other braces name a parameter.

It has not yet run to the end. Its first run failed in `create()` (above), and NEST runs only through ci-beNNch, on the
cluster.
