"""E-prop with biological features, profiled: NEST 3.10's event-based eprop_iaf and eprop_synapse as a ci-beNNch benchmark.

**Purpose.** One number: the share f of a plastic run's cycles that NEST spends in its per-synapse gradient loop
(compute_gradient). f decides the constant-time plasticity project, as registered in
sp4-direction-scoping/tier2_project_2026-11-06/REGISTRATION.md. This is a profiling workload, not a model of a brain
area.

**The network** follows the scaling setup of Korcsak-Gorzo and others (arXiv 2511.21674, section 5.8) at a size that
fits one node. The neuron parameters are those of NEST's own e-prop example
(pynest/examples/eprop_plasticity/eprop_supervised_regression_sine-waves.py):
- **inputs:** N_IN parrot neurons, fed by one poisson_generator at RATE_IN spikes/s, with an independent train for each
  parrot;
- **recurrent:** N_REC eprop_iaf neurons, each spiking exactly every INTERVAL ms through NEST's ignore_and_spike
  mechanism, at offsets drawn uniformly from 1 to INTERVAL ms. So every recurrent inter-spike interval is INTERVAL
  steps, and nothing in the profile depends on the network's dynamics;
- **connectivity:** input in-degree K_IN and recurrent in-degree K_REC, with no autapses or multapses. N_OUT
  eprop_readout neurons each take K_OUT recurrent neurons, and each readout sends learning signals to FEEDBACK_OUT
  recurrent neurons;
- **time:** 1 ms resolution, and every delay 1 ms (one step).

**PLASTICITY** selects the synapses of the three plastic projections (input, recurrent and readout):
- "eprop_gd": eprop_synapse, with gradient descent applied once per spike, as in NEST's event-based examples;
- "eprop_adam": the same with Adam;
- "static": static_synapse, with weights from the same distributions.

The trace cutoff is CUTOFF ms, NEST's examples' value and equal to INTERVAL.

**How it runs.**
- The warm-up and the measured period run with nest.Prepare and nest.Run, as in this repository's Brunel benchmark
  (branch brunel), whose logging is repeated here.
- perf_recording.py, copied from branch save-cycle-log-perf with MODE "events", records ranks 0 and 1 during the
  measured period only. Its sampling periods are set below from the config's perf_period.
- perf_shares.py then reads each recording on the node and writes perf_<rank>_shares.json.
- logging() runs after perf's write-out, so a write-out that overruns the walltime loses the logfile. The walltime
  is set to cover it.
- JUBE replaces the eight placeholders below. No other braces in this file may name a parameter.
"""
import os
import resource
import time

import nest
import numpy as np

import perf_recording
from perf_recording import Recording, write_pages
from perf_shares import write_shares

T_PRESIM = {model_time_presim}  # ms of warm-up, excluded from the timers below
T_SIM = {model_time_sim}  # ms simulated and measured, perf recording on ranks 0 and 1
THREADS = {threads_per_task}  # per rank
SEED = {rng_seed}
PLASTICITY = "{plasticity}"  # "eprop_gd", "eprop_adam" or "static"
N_REC = {n_rec}  # recurrent eprop_iaf neurons
K_REC = {k_rec}  # recurrent in-degree
PERF_PERIOD = {perf_period}  # cycles per perf sample; instructions are sampled at a third of it

perf_recording.ROLES = ((("cycles",), PERF_PERIOD), (("instructions",), PERF_PERIOD // 3))

DT = 1.0  # ms, resolution, as in NEST's e-prop examples and the scaling setup
DELAY = 1.0  # ms, every connection's delay (one step)
N_IN = 1000  # input parrot neurons
RATE_IN = 5.0  # spikes/s of each input
K_IN = 100  # input in-degree of a recurrent neuron
N_OUT = 10  # readout neurons
K_OUT = 1000  # in-degree of a readout neuron (at most N_REC)
FEEDBACK_OUT = 100  # recurrent neurons each readout sends its learning signal to (at most N_REC)
INTERVAL = 100.0  # ms between a recurrent neuron's spikes (ignore_and_spike): 10 spikes/s
CUTOFF = 100.0  # ms, eprop_isi_trace_cutoff, the examples' value
KAPPA = 0.97  # the examples' low-pass filter of the eligibility trace
SCALE = 1.0 - KAPPA  # the examples' weight scale for eprop_iaf

# the sine-wave example's recurrent and readout parameters, with the fixed spiking added
PARAMS_REC = dict(C_m=1.0, c_reg=300.0 / 1000.0, E_L=0.0, eprop_isi_trace_cutoff=CUTOFF, f_target=10.0,
                  flush_event_send_interval=1000.0, I_e=0.0, kappa=KAPPA, kappa_reg=0.97,
                  surrogate_gradient_function="piecewise_linear", surrogate_gradient_height=10.0,
                  surrogate_gradient_width=1.0 / 33.3, t_ref=0.0, tau_m=30.0, V_m=0.0, V_th=0.03,
                  ignore_and_spike=True, ignore_and_spike_interval=INTERVAL)
PARAMS_OUT = dict(C_m=1.0, E_L=0.0, eprop_isi_trace_cutoff=CUTOFF, I_e=0.0, tau_m=30.0, V_m=0.0)
OPTIMIZER = dict(type="adam" if PLASTICITY == "eprop_adam" else "gradient_descent", batch_size=1,
                 eta=1e-4 * SCALE ** 2, optimize_each_step=False, Wmin=-100.0, Wmax=100.0)


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
    """bm_helpers.logging of the PD14 benchmark, as in the Brunel benchmark: the kernel status with the warm-up
    subtracted from the simulation timers, the Python timers, the memory, this rank's peak resident memory, and the
    whole per-cycle log."""
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
        f.write('plasticity ' + PLASTICITY + '\n')
    # NEST built with cycle timers keeps one entry per cycle in its cycle_time_log; the logfile shortens long arrays,
    # so each rank also saves the whole log, one array per key.
    if 'cycle_time_log' in d:
        np.savez(os.path.join('data', 'cycle_time_log_' + str(nest.Rank())),
                 **dict((key, np.asarray(value)) for key, value in d['cycle_time_log'].items()))


def synapses(std):
    """The plastic projections' synapse: eprop_synapse or static_synapse, normal weights of mean 0."""
    model = "static_synapse" if PLASTICITY == "static" else "eprop_synapse"
    return dict(synapse_model=model, delay=DELAY, weight=nest.random.normal(mean=0.0, std=std))


py_timers = dict()
memory_used = dict()
os.makedirs('data', exist_ok=True)
recordings = os.path.join('data', 'eprop', 'recordings')   # matches the fetch rules for perf_* and pages_*
os.makedirs(recordings, exist_ok=True)

t0 = time.time()
nest.ResetKernel()
nest.set(resolution=DT, local_num_threads=THREADS, rng_seed=SEED, print_time=False)
memory_used['base_memory'] = memory()
t1 = time.time()
gen_in = nest.Create("poisson_generator", params=dict(rate=RATE_IN))
nrns_in = nest.Create("parrot_neuron", N_IN)
nrns_rec = nest.Create("eprop_iaf", N_REC, PARAMS_REC)
nrns_out = nest.Create("eprop_readout", N_OUT, PARAMS_OUT)
gen_target = nest.Create("step_rate_generator", N_OUT, params=dict(amplitude_times=[2.0], amplitude_values=[0.0]))
gen_window = nest.Create("step_rate_generator", 1, params=dict(amplitude_times=[2.0], amplitude_values=[1.0]))
# offsets of 1 to INTERVAL ms, drawn by NEST's own random parameters, which every rank evaluates for its own neurons
nrns_rec.set(ignore_and_spike_offset=nest.random.uniform_int(int(INTERVAL)) + 1.0)
t2 = time.time()
memory_used['node_memory'] = memory()
if PLASTICITY != "static":
    nest.SetDefaults("eprop_synapse", dict(optimizer=OPTIMIZER))
nest.Connect(gen_in, nrns_in, "all_to_all", dict(synapse_model="static_synapse", delay=DELAY))
nest.Connect(nrns_in, nrns_rec, dict(rule="fixed_indegree", indegree=K_IN), synapses(SCALE / np.sqrt(K_IN)))
nest.Connect(nrns_rec, nrns_rec,
             dict(rule="fixed_indegree", indegree=K_REC, allow_autapses=False, allow_multapses=False),
             synapses(SCALE / np.sqrt(K_REC)))
nest.Connect(nrns_rec, nrns_out, dict(rule="fixed_indegree", indegree=K_OUT, allow_multapses=False),
             synapses(SCALE / np.sqrt(K_OUT)))
nest.Connect(nrns_out, nrns_rec, dict(rule="fixed_outdegree", outdegree=FEEDBACK_OUT, allow_multapses=False),
             dict(synapse_model="eprop_learning_signal_connection", delay=DELAY,
                  weight=nest.random.normal(mean=0.0, std=SCALE / np.sqrt(N_OUT))))
nest.Connect(gen_target, nrns_out, "one_to_one",
             dict(synapse_model="rate_connection_delayed", delay=DELAY, receptor_type=2))
nest.Connect(gen_window, nrns_out, "all_to_all",
             dict(synapse_model="rate_connection_delayed", delay=DELAY, receptor_type=1))
t3 = time.time()
memory_used['network_memory'] = memory()
nest.Prepare()
t4 = time.time()
nest.Run(T_PRESIM)
t5 = time.time()
memory_used['init_memory'] = memory()
intermediate_kernel_status = nest.GetKernelStatus()
write_pages(os.path.join(recordings, 'pages_' + str(nest.Rank()) + '_start.txt'))
recording = Recording(nest.Rank(), recordings).start()
t5r = time.time()
nest.Run(T_SIM)
t6 = time.time()
recording.stop()
write_pages(os.path.join(recordings, 'pages_' + str(nest.Rank()) + '_end.txt'))
memory_used['total_memory'] = memory()

py_timers['py_time_kernel_prepare'] = t1 - t0
py_timers['py_time_create'] = t2 - t1
py_timers['py_time_connect'] = t3 - t2
py_timers['py_time_network_prepare'] = t4 - t3
py_timers['py_time_presimulate'] = t5 - t4
py_timers['py_time_perf_start'] = t5r - t5
py_timers['py_time_simulate'] = t6 - t5r
print('Rank', nest.Rank(), PLASTICITY, py_timers, memory_used)
logging(py_timers, memory_used, intermediate_kernel_status)
write_shares(recordings, nest.Rank())
nest.Cleanup()
