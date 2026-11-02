# Brunel's network as a ci-beNNch benchmark

Brunel (2000), model A, in its asynchronous irregular state. The network is NEST 3.10's own example,
`pynest/examples/brunel_delta_nest.py` (GPL-2.0-or-later, as NEST), with its numbers and calls unchanged:
- 10,000 excitatory and 2,500 inhibitory `iaf_psc_delta` units, every V_m starting at 0;
- `fixed_indegree` 1,000 from the excitatory and 250 from the inhibitory population onto every unit, with NEST's
  defaults (autapses and multapses allowed);
- weights 0.1 and −0.5 mV, and every delay 1.5 ms;
- one `poisson_generator` at 20,000 events per second onto every unit.

## Files

| File | What it is |
|---|---|
| `run_benchmark.py` | The example's network, timed step by step. It then runs the warm-up and the measured period with `nest.Prepare` and `nest.Run`, as the multi-area model's benchmark does. |

## What a run records

Each rank writes two files into `data/`, as the PD14 benchmark does, so ci-beNNch's `collect_timer_data.py` reads
them unchanged:
- `logfile_<rank>`: the kernel status with the warm-up subtracted from the simulation timers, the Python timers, NEST's
  memory (virtual size) and the rank's peak resident memory;
- `cycle_time_log_<rank>.npz`: NEST's whole per-cycle log.

The seed, the threads and the durations come from the ci-beNNch config: 500 ms of warm-up, then 1,000 ms measured. The
example's two recorders (50 units each) and its plot are left out, as the multi-area model's benchmark has no recording
devices.

**One difference from the other benchmarks.** Every delay is 1.5 ms, so NEST's cycle is 15 steps. The 1,000 ms
measured are 667 cycles, not 10,000, and per-cycle readings are over 15 steps.

## Checked before any run

- `run_benchmark.py` parses once its four placeholders are filled. No other braces name a parameter.
- Every call exists in NEST 3.10: `nest.set` (`pynest/nest/__init__.py:124`), `Create`, `CopyModel`, `Connect`,
  `Prepare`, `Run`, `Cleanup`.
- The network section is the example's, line by line, apart from the recorders and its output settings
  (`print_time`, `overwrite_files`).

It has not been run: NEST runs only through ci-beNNch, on the cluster.
