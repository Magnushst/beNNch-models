"""The human multi-area model's statistics run: run_benchmark.py with the model's own event recorder on. NOT A REFERENCE
RUN: recording adds work, so its timings are not NEST's reference; it exists to compare activity population by
population (config branch MagnusStromme-stats).

As run_benchmark.py, the human multi-area model (Pronold and others 2024), ground state, as a ci-beNNch benchmark.

The model is built and simulated by its own simulation.py (INM-6/human-multi-area-model at cab39b37), from the
network its own code built (network.json, network_json.py). This script runs simulation.py's setup steps in the order
of its Simulation.setup, with a timer around each, then simulates the warm-up and the measured period with
nest.Prepare and nest.Run, as the multi-area model's benchmark does. Each rank writes its logfile and its per-cycle log
into data/, as PD14's benchmark does (bm_helpers.py), for ci-beNNch's collect_timer_data.py.

The ground-state experiment (experiments/experiment_stabilized_groundstate.py) sets the seed and the duration; the
benchmark sets both. Here simulation.py's own recorder records every unit's events (rec_dev ['spike_recorder']: one
text file per virtual process in data/spikes/, as the model writes them). JUBE replaces the four placeholders below; no
other braces in this file may name a parameter.
"""
import os
import resource
import time

import nest
import numpy as np

from default_sim_params import params as sim_dict
from network_json import network_dict
from simulation import Simulation

T_PRESIM = {model_time_presim}  # ms of warm-up, excluded from the timers below (the model's analysis drops 500 ms)
T_SIM = {model_time_sim}  # ms simulated and measured
THREADS = {threads_per_task}  # per rank
sim_dict.update(master_seed={rng_seed}, rec_dev=['spike_recorder'])


def memory():
    """NEST's memory_size, as the benchmark's other models record it (bm_helpers.py): the virtual size in KiB."""
    try:
        mem = nest.get("memory_size")
    except KeyError:
        mem = nest.ll_api.sli_func('memory_thisjob')
    if isinstance(mem, dict):
        return mem['heap']
    return mem


def logging(py_timers, memory_used, intermediate_kernel_status):
    """bm_helpers.logging of the PD14 benchmark: the kernel status with the warm-up subtracted from the simulation
    timers, the Python timers, the memory, this rank's peak resident memory, and the whole per-cycle log."""
    d = nest.GetKernelStatus()
    presim_timers = ['time_collocate_spike_data', 'time_communicate_spike_data', 'time_deliver_secondary_data',
                     'time_deliver_spike_data', 'time_gather_secondary_data', 'time_gather_spike_data',
                     'time_omp_synchronization_simulation', 'time_mpi_synchronization', 'time_simulate', 'time_update']
    presim_timers.extend([timer + '_cpu' for timer in presim_timers])
    other_timers = ['time_communicate_prepare', 'time_communicate_target_data', 'time_construction_connect',
                    'time_construction_create', 'time_gather_target_data', 'time_omp_synchronization_construction']
    other_timers.extend([timer + '_cpu' for timer in other_timers])
    for timer in presim_timers:
        try:
            try:
                timer_array = tuple(d[timer][tid] - intermediate_kernel_status[timer][tid] for tid in range(len(d[timer])))
                d[timer] = timer_array[0]
                d[timer + "_max"] = max(timer_array)
                d[timer + "_min"] = min(timer_array)
                d[timer + "_mean"] = np.mean(timer_array)
                d[timer + "_all"] = timer_array
                d[timer + '_presim'] = intermediate_kernel_status[timer][0]
                d[timer + "_presim_max"] = max(intermediate_kernel_status[timer])
                d[timer + "_presim_min"] = min(intermediate_kernel_status[timer])
                d[timer + "_presim_avg"] = np.mean(intermediate_kernel_status[timer])
                d[timer + "_presim_all"] = intermediate_kernel_status[timer]
            except TypeError:
                # No threaded timers, fall back to scalar handling
                d[timer] -= intermediate_kernel_status[timer]
                d[timer + '_presim'] = intermediate_kernel_status[timer]
        except KeyError:
            # KeyError if compiled without detailed timers, except time_simulate
            continue
    for timer in other_timers:
        try:
            try:
                timer_array = d[timer]
                d[timer] = timer_array[0]
                d[timer + "_max"] = max(timer_array)
                d[timer + "_min"] = min(timer_array)
                d[timer + "_mean"] = np.mean(timer_array)
                d[timer + "_all"] = timer_array
            except TypeError:
                # No threaded timers, d[timer] is already a scalar
                continue
        except KeyError:
            continue
    with open(os.path.join('data', 'logfile_' + str(nest.Rank())), 'w') as f:
        for key, val in d.items():
            f.write(key + ' ' + str(val) + '\n')
        for key, value in py_timers.items():
            f.write(key + ' ' + str(value) + '\n')
        for key, value in memory_used.items():
            f.write(key + ' ' + str(value) + '\n')
        # The memory above is NEST's virtual size; also the most memory this rank has held, its peak resident set in
        # KiB as the kernel counts it.
        f.write('peak_resident_memory_kib ' + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) + '\n')
    # NEST built with cycle timers keeps one entry per cycle in its cycle_time_log; the logfile shortens long arrays,
    # so each rank also saves the whole log, one array per key.
    if 'cycle_time_log' in d:
        np.savez(os.path.join('data', 'cycle_time_log_' + str(nest.Rank())),
                 **dict((key, np.asarray(value)) for key, value in d['cycle_time_log'].items()))


py_timers = dict()
memory_used = dict()
memory_used['base_memory'] = memory()
os.makedirs(os.path.join('data', 'spikes'), exist_ok=True)  # every rank writes here; simulation.py creates it on rank 0 only

sim = Simulation(sim_dict, network_dict())
t0 = time.time()
sim.set_data_path(os.path.join(os.getcwd(), 'data'))
sim.setup_nest(THREADS)
t1 = time.time()
sim.create_populations()
sim.create_devices()
sim.create_poisson()
sim.create_single_spike()
t2 = time.time()
memory_used['node_memory'] = memory()
sim.connect_neurons()
sim.connect_poisson()
sim.connect_single_spike()
sim.connect_devices()
t3 = time.time()
memory_used['network_memory'] = memory()
nest.Prepare()
t4 = time.time()
nest.Run(T_PRESIM)
t5 = time.time()
memory_used['init_memory'] = memory()
intermediate_kernel_status = nest.GetKernelStatus()
nest.Run(T_SIM)
t6 = time.time()
memory_used['total_memory'] = memory()

py_timers['py_time_kernel_prepare'] = t1 - t0
py_timers['py_time_create'] = t2 - t1
py_timers['py_time_connect'] = t3 - t2
py_timers['py_time_network_prepare'] = t4 - t3
py_timers['py_time_presimulate'] = t5 - t4
py_timers['py_time_simulate'] = t6 - t5
print('Rank', nest.Rank(), py_timers, memory_used)
logging(py_timers, memory_used, intermediate_kernel_status)
nest.Cleanup()
