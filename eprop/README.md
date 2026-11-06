# E-prop profile: NEST 3.10's event-based e-prop as a ci-beNNch benchmark

A profiling workload, not a model of a brain area. It measures one number: the share f of a plastic run's cycles in
NEST's per-synapse gradient loop (`compute_gradient`). The registration with its decision rule is
`sp4-direction-scoping/tier2_project_2026-11-06/REGISTRATION.md`, and the derivation it tests is `DERIVATION.md`
there.

**Files:**
- `run_benchmark.py`: the network, the warm-up and the measured period, and the Brunel benchmark's logging. Its
  docstring gives the network.
- `perf_recording.py`: the model hook of branch save-cycle-log-perf (e10d03d), copied with MODE "events". It records
  ranks 0 and 1 during the measured period only.
- `perf_shares.py`: the registered reading, run on the node after the recording. It writes perf_<rank>_shares.json
  into data/eprop/recordings/, where the existing fetch rules for perf_* and pages_* match.

**Parameters** (set by the config, branch MagnusStromme-eprop):
- `model_time_presim` and `model_time_sim`, in ms;
- `rng_seed`;
- `plasticity`: eprop_gd, eprop_adam or static;
- `n_rec` and `k_rec`, the recurrent neurons and their in-degree.

**Fixed in the script:**
- 1,000 Poisson inputs at 5 spikes/s, with input in-degree 100;
- 10 readouts with in-degree 1,000 and feedback out-degree 100;
- every recurrent neuron spiking every 100 ms (ignore_and_spike);
- a trace cutoff of 100 ms;
- 1 ms resolution and 1 ms delays.

**Origins:**
- **The structure:** the scaling setup of Korcsak-Gorzo and others (arXiv 2511.21674, section 5.8), at a size that
  fits one node.
- **The neuron parameters:** NEST's `pynest/examples/eprop_plasticity/eprop_supervised_regression_sine-waves.py`.
- **The optimiser:** gradient descent, or Adam, once per spike, as in NEST's event-based examples.

**Status:** not run. NEST is the stock reference branch, v3.10-cycle-phase-timers, unchanged.
