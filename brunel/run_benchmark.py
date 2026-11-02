"""Brunel's network (Brunel 2000, model A, asynchronous irregular state) as a ci-beNNch benchmark.

The network is NEST 3.10's own example pynest/examples/brunel_delta_nest.py, its numbers and calls unchanged:
10,000 excitatory and 2,500 inhibitory iaf_psc_delta units, fixed_indegree 1,000 and 250 (NEST's defaults: autapses
and multapses allowed), weights 0.1 and -0.5 mV, every delay 1.5 ms, one poisson_generator at 20,000 events per second
onto every unit. The example's recorders and plotting are left out, as the multi-area model's benchmark has no
recording devices, so only the seed, the threads and the durations are the benchmark's. The warm-up and the measured
period run with nest.Prepare and nest.Run, as in the multi-area model's benchmark. Each rank writes its logfile and its
per-cycle log into data/, as PD14's benchmark does (bm_helpers.py), for ci-beNNch's collect_timer_data.py.

With every delay 1.5 ms, NEST's cycle is 15 steps, so 1,000 ms measured are 667 cycles, not 10,000. JUBE replaces the
four placeholders below; no other braces in this file may name a parameter.
"""
import os
import resource
import time

import nest
import numpy as np

T_PRESIM = {model_time_presim}  # ms of warm-up, excluded from the timers below
T_SIM = {model_time_sim}  # ms simulated and measured
THREADS = {threads_per_task}  # per rank
SEED = {rng_seed}

# brunel_delta_nest.py's parameters
dt = 0.1  # the resolution in ms
delay = 1.5  # synaptic delay in ms
g = 5.0  # ratio inhibitory weight/excitatory weight
eta = 2.0  # external rate relative to threshold rate
epsilon = 0.1  # connection probability
order = 2500
NE = 4 * order  # number of excitatory neurons
NI = 1 * order  # number of inhibitory neurons
CE = int(epsilon * NE)  # number of excitatory synapses per neuron
CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
tauMem = 20.0  # time constant of membrane potential in ms
theta = 20.0  # membrane threshold potential in mV
neuron_params = dict(C_m=1.0, tau_m=tauMem, t_ref=2.0, E_L=0.0, V_reset=0.0, V_m=0.0, V_th=theta)
J = 0.1  # postsynaptic amplitude in mV
J_ex = J  # amplitude of excitatory postsynaptic potential
J_in = -g * J_ex  # amplitude of inhibitory postsynaptic potential
nu_th = theta / (J * CE * tauMem)
nu_ex = eta * nu_th
p_rate = 1000.0 * nu_ex * CE


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
os.makedirs('data', exist_ok=True)

t0 = time.time()
nest.ResetKernel()
nest.set(resolution=dt, local_num_threads=THREADS, rng_seed=SEED, print_time=False)
memory_used['base_memory'] = memory()
t1 = time.time()
nodes_ex = nest.Create("iaf_psc_delta", NE, params=neuron_params)
nodes_in = nest.Create("iaf_psc_delta", NI, params=neuron_params)
noise = nest.Create("poisson_generator", params=dict(rate=p_rate))
t2 = time.time()
memory_used['node_memory'] = memory()
nest.CopyModel("static_synapse", "excitatory", dict(weight=J_ex, delay=delay))
nest.CopyModel("static_synapse", "inhibitory", dict(weight=J_in, delay=delay))
nest.Connect(noise, nodes_ex, syn_spec="excitatory")
nest.Connect(noise, nodes_in, syn_spec="excitatory")
nest.Connect(nodes_ex, nodes_ex + nodes_in, dict(rule="fixed_indegree", indegree=CE), "excitatory")
nest.Connect(nodes_in, nodes_ex + nodes_in, dict(rule="fixed_indegree", indegree=CI), "inhibitory")
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
