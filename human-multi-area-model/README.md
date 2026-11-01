# The human multi-area model as a ci-beNNch benchmark

The human multi-area model of Pronold and others (2024) is 34 cortical areas, 254 populations, 3,468,839 units and
42,778,075,138 connections, here in its ground state with default parameters. It is built and simulated by its own code,
INM-6/human-multi-area-model at commit `cab39b3701a35131539f41c6aed767b4e87bd8df` (GPL-3.0, `LICENSE`).

## Files

| File | What it is |
|---|---|
| `simulation.py` | The model's `src/simulation.py`. Only change: the `yaml` and `dicthash` imports move into the three functions that use them, none of which the benchmark calls (header comment). |
| `default_sim_params.py`, `helpers/lognormal.py`, `LICENSE` | The model's own files, unchanged (md5 equal to the commit's). |
| `network.json` | The network the model's own code built (`src/snakemake_network.py` with `experiments/experiment_stabilized_groundstate.py`, dump `49f3db1e4376985ef1aaf4f235cabeba`). It holds every value `simulation.py` reads, plus the md5 of every dump file. |
| `make_network_json.py` | Writes `network.json` from the dump. It needs pandas 2.2.3, so it runs once, off the cluster. |
| `network_json.py` | Rebuilds the dictionary `simulation.py` reads from `network.json`, answering its `.items()` and `.loc` calls without pandas. |
| `run_benchmark.py` | Runs `simulation.py`'s setup steps in the order of its `Simulation.setup`, timing each one. It then runs the warm-up and the measured period with `nest.Prepare` and `nest.Run`, as the multi-area model's benchmark does. |

## What a run records

Each rank writes two files into `data/`, as the PD14 benchmark does, so ci-beNNch's `collect_timer_data.py` reads
them unchanged:
- `logfile_<rank>`: the kernel status with the warm-up subtracted from the simulation timers, the Python timers, NEST's
  memory (virtual size) and the rank's peak resident memory;
- `cycle_time_log_<rank>.npz`: NEST's whole per-cycle log.

The seed and the durations come from the ci-beNNch config: 500 ms of warm-up, the length the model's own analysis drops,
then 1,000 ms measured. No recording devices are used, as in the multi-area model's benchmark.

## Checked before any run

The values `network_json.py` returns were checked against the model's pickles read through pandas, at all 85,442
lookups `simulation.py` makes: integers as integers, floats bit-equal. The check ran neither NEST nor the model's
code. It is `check_nest_human_network.py` in the minimal-state prototype, with its record `NEST_HUMAN_NETWORK_CHECK.json`.
