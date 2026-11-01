"""Writes network.json: every value of the human multi-area model's network that simulation.py reads, as plain JSON,
so that the benchmark needs neither pandas nor the model's pickles.
    <python with pandas 2.2.3 and pyyaml> make_network_json.py <dump folder>

The dump folder is what the model's own code writes (github.com/INM-6/human-multi-area-model at commit cab39b37,
src/snakemake_network.py with experiments/experiment_stabilized_groundstate.py, the ground state with default
parameters): pandas pickles and net.yaml, read here as the model's networkDictFromDump reads them. network.json keeps
- "yaml": the scalars and dicts simulation.py reads from net.yaml;
- "series": each Series simulation.py reads, as [index, value] pairs in the dump's order;
- "frames": synapses_internal, weights and weights_sd at every pair of populations with units that simulation.py
  reads (the first at every such pair with a nonzero count, the other two where the count is nonzero), and delay_cc
  and delay_cc_sd at every pair of areas;
- "provenance": the repository, commit and experiment, and the md5 of every file in the dump.
Integers stay integers and floats keep every bit (JSON writes them with repr). A NaN (spike_time: no single spike) is
written as null. network_json.py turns this back into the dictionary simulation.py expects.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
YAML_KEYS = ['neuron_model_E', 'neuron_model_I', 'neuron_params_E', 'neuron_params_I', 'neuron_param_dist_E',
             'neuron_param_dist_I', 'delay_e', 'delay_e_sd', 'delay_i', 'delay_i_sd', 'delay_distribution', 'p_transmit']
SERIES = ['neuron_numbers', 'dc_drive', 'synapses_external', 'rate_ext', 'spike_time', 'weights_ext', 'weights_ext_sd']


def plain(v):
    """A pandas or numpy scalar as a JSON value of the same kind; NaN as None."""
    if isinstance(v, (bool, str)) or v is None:
        return v
    if hasattr(v, 'dtype') and v.dtype.kind in 'iu' or isinstance(v, int):
        return int(v)
    v = float(v)
    return None if math.isnan(v) else v


def main(dump):
    dump = Path(dump)
    with open(dump / 'net.yaml') as f:
        net = yaml.load(f, Loader=yaml.Loader)
    pk = {f.stem: pd.read_pickle(f) for f in sorted(dump.glob('*.pkl'))}
    nn = pk['neuron_numbers']
    keys = list(nn.index)
    pops = [k for k, n in nn.items() if n > 0]
    for name in SERIES:
        assert list(pk[name].index) == keys, name
    si, w, wsd = pk['synapses_internal'], pk['weights'], pk['weights_sd']
    index = {k: i for i, k in enumerate(keys)}
    counts, weights, weights_sd = [], [], []
    for t in pops:
        for s in pops:
            count = si.loc[t, s]
            if count > 0:
                counts.append([index[t], index[s], plain(count)])
                weights.append([index[t], index[s], plain(w.loc[t, s])])
                weights_sd.append([index[t], index[s], plain(wsd.loc[t, s])])
    dcc, dcc_sd = pk['delay_cc'], pk['delay_cc_sd']
    areas = list(dcc.index)
    assert list(dcc.columns) == areas and list(dcc_sd.index) == areas and list(dcc_sd.columns) == areas
    out = {
        'provenance': {
            'repository': 'https://github.com/INM-6/human-multi-area-model',
            'commit': 'cab39b3701a35131539f41c6aed767b4e87bd8df',
            'experiment': 'experiments/experiment_stabilized_groundstate.py',
            'dump': dump.name,
            'files_md5': {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in sorted(dump.iterdir()) if f.is_file()},
        },
        'yaml': {k: net[k] for k in YAML_KEYS},
        'populations': [list(k) for k in keys],
        'series': {name: [plain(v) for v in pk[name].values] for name in SERIES},
        'frames': {
            'synapses_internal': counts, 'weights': weights, 'weights_sd': weights_sd,
            'areas': areas,
            'delay_cc': [[plain(v) for v in row] for row in dcc.values],
            'delay_cc_sd': [[plain(v) for v in row] for row in dcc_sd.values],
        },
    }
    (HERE / 'network.json').write_text(json.dumps(out, indent=0, allow_nan=False) + '\n')
    print(len(keys), 'populations,', len(pops), 'with units,', len(counts), 'pathways,',
          sum(c[2] for c in counts), 'connections')


if __name__ == '__main__':
    main(sys.argv[1])
