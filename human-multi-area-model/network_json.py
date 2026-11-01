"""The network dictionary simulation.py reads, rebuilt from network.json (make_network_json.py) without pandas.

simulation.py reads the model's Series and DataFrames only through .items() and .loc; Series and Frame answer exactly
those calls with the dump's values, in the dump's order. A pair of populations absent from synapses_internal has no
connections (0), as in the dump; weights and weights_sd hold the pairs simulation.py reads, those with connections.
"""
import json
import math
import os

NETWORK_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'network.json')


class _Loc:
    def __init__(self, values, missing):
        self._values = values
        self._missing = missing

    def __getitem__(self, key):
        if self._missing is None:
            return self._values[key]
        return self._values.get(key, self._missing)


class Series:
    """A pandas Series as simulation.py uses it: .items() in the dump's order, and .loc[index]."""

    def __init__(self, index, values):
        self._items = list(zip(index, values))
        self.loc = _Loc(dict(self._items), None)

    def items(self):
        return iter(self._items)


class Frame:
    """A pandas DataFrame as simulation.py uses it: .loc[row, column]."""

    def __init__(self, values, missing=None):
        self.loc = _Loc(values, missing)


def network_dict(path=NETWORK_JSON):
    with open(path) as f:
        d = json.load(f)
    keys = [tuple(k) for k in d['populations']]
    net = dict(d['yaml'])
    for name, values in d['series'].items():
        net[name] = Series(keys, [math.nan if v is None else v for v in values])
    frames = d['frames']

    def pairs(entries):
        return {(keys[t], keys[s]): v for t, s, v in entries}

    net['synapses_internal'] = Frame(pairs(frames['synapses_internal']), missing=0)
    net['weights'] = Frame(pairs(frames['weights']))
    net['weights_sd'] = Frame(pairs(frames['weights_sd']))
    areas = frames['areas']
    for name in ('delay_cc', 'delay_cc_sd'):
        net[name] = Frame({(a, b): frames[name][i][j] for i, a in enumerate(areas) for j, b in enumerate(areas)})
    return net
