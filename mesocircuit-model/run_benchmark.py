"""The mesocircuit of Senk and others (Cerebral Cortex 2024), parameter set mesocircuit_MAMV1, as a ci-beNNch benchmark.

The model is built by its own code, INM-6/mesocircuit-model at commit f59ada6b3c3efb1fa18f96c638dd1741ebfb77e5
(GPL-3.0, LICENSE.rst), copied unchanged into mesocircuit/ and parametersets.py. The parameters are derived as the
model's mesocircuit_framework.py derives them: the base dictionaries, the parameter set's net_dict merged in by
helpers.merge_dictionaries (repeated below, as helpers.py needs the 'parameters' package), then
helpers_network.derive_dependent_parameters. The network is built by the model's Network class (network.py) with
create() and connect(), as its run_network.py does; connect() ends with nest.Prepare() and nest.Cleanup(). The
warm-up and the measured period then run with nest.Prepare and nest.Run, as the multi-area model's benchmark does.
Each rank writes its logfile and its per-cycle log into data/, as PD14's benchmark does (bm_helpers.py), for
ci-beNNch's collect_timer_data.py.

The benchmark sets the seed, the threads and the durations. No recording devices (rec_dev empty), as in the
multi-area model's benchmark. The model's create() still writes its node ids and unit positions into
model_output/raw_data/. JUBE replaces the four placeholders below; no other braces in this file may name a parameter.
"""
import copy
import os
import resource
import time

import nest
import numpy as np

import parametersets
from mesocircuit.parameterization import base_network_params, base_simulation_params
from mesocircuit.parameterization import helpers_network as helpnet
from mesocircuit.simulation import network

T_PRESIM = {model_time_presim}  # ms of warm-up, excluded from the timers below
T_SIM = {model_time_sim}  # ms simulated and measured
THREADS = {threads_per_task}  # per rank
SEED = {rng_seed}
PARAMETER_SET = 'mesocircuit_MAMV1'


def merge_dictionaries(main_dict, new_dict):
    """mesocircuit/helpers/helpers.py merge_dictionaries: merges new_dict recursively into main_dict."""
    for key in new_dict:
        if key in main_dict and isinstance(main_dict[key], dict) and isinstance(new_dict[key], dict):
            merge_dictionaries(main_dict[key], new_dict[key])
        else:
            main_dict[key] = new_dict[key]
    return main_dict


class Circuit:
    """What Network reads of a mesocircuit_framework.Mesocircuit object."""

    def __init__(self):
        custom = parametersets.ps_dicts[PARAMETER_SET]
        net_dict = merge_dictionaries(copy.deepcopy(base_network_params.net_dict), copy.deepcopy(custom['net_dict']))
        self.net_dict = helpnet.derive_dependent_parameters(net_dict)
        self.sim_dict = merge_dictionaries(copy.deepcopy(base_simulation_params.sim_dict),
                                           copy.deepcopy(custom.get('sim_dict', {})))
        self.sim_dict.update(rng_seed=SEED, rec_dev=[], print_time=False)
        self.data_dir_circuit = os.path.join(os.getcwd(), 'model_output')


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
os.makedirs('data', exist_ok=True)  # every rank writes here
circuit = Circuit()
os.makedirs(os.path.join(circuit.data_dir_circuit, 'raw_data'), exist_ok=True)

t0 = time.time()
net = network.Network(circuit, THREADS)  # resets the kernel and sets threads, resolution and seed
memory_used['base_memory'] = memory()
t1 = time.time()
net.create()
t2 = time.time()
memory_used['node_memory'] = memory()
net.connect()
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
