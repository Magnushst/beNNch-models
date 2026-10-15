import json
import os

import nest
import numpy as np


def logging(py_timers=None, memory_used=None, intermediate_kernel_status={}):
    """
    Write runtime and memory for all MPI processes to file.
    """
    fn = os.path.join('data',
                      '_'.join(('logfile',
                                str(nest.Rank()))))
    d = nest.GetKernelStatus()

    presim_timers = ['time_collocate_spike_data', 'time_communicate_spike_data', 'time_deliver_secondary_data', 'time_deliver_spike_data', 'time_gather_secondary_data', 'time_gather_spike_data', 'time_omp_synchronization_simulation', 'time_mpi_synchronization', 'time_simulate', 'time_update']
    presim_timers.extend([timer + '_cpu' for timer in presim_timers])
    other_timers = ['time_communicate_prepare', 'time_communicate_target_data', 'time_construction_connect', 'time_construction_create', 'time_gather_target_data', 'time_omp_synchronization_construction']
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
                # No threaded timers, d[timer] is already a scalar and is set after nest.kernel_status
                continue
        except KeyError:
            # KeyError if compiled without detailed timers, except time_simulate
            continue
    
    with open(fn, 'w') as f:
        for key, val in d.items():
            f.write(key + ' ' + str(val) + '\n')
        if py_timers:
            for key, value in py_timers.items():
                f.write(key + ' ' + str(value) + '\n')
        if memory_used:
            for key, value in memory_used.items():
                f.write(key + ' ' + str(value) + '\n')

    # NEST built with cycle timers keeps one entry per cycle in its cycle_time_log. The logfile above
    # shortens long arrays, so each rank also saves the whole log, one array per key.
    if 'cycle_time_log' in d:
        np.savez(os.path.join('data', '_'.join(('cycle_time_log', str(nest.Rank())))),
                 **{key: np.asarray(value) for key, value in d['cycle_time_log'].items()})


def memory():
    """
    Use NEST's memory wrapper function to record used memory.
    """
    try:
        mem = nest.get("memory_size")
    except KeyError:
        try:
            mem = nest.ll_api.sli_func('memory_thisjob')
        except AttributeError:
            mem = nest.sli_func('memory_thisjob')
    if isinstance(mem, dict):
        return mem['heap']
    else:
        return mem
