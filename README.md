# Spiking-network models

[Brian2](https://brian2.readthedocs.io) implementations of the two
spiking-network models of the paper, with one script per figure.

| File | Contents |
|---|---|
| `models.py` | `TwoLayerModel` (two-layer model) and `MultiAssemblyModel` (four-module model) |
| `two_layer.py` | Fig. 4 B–D: two-layer model for three inter-layer connectivities |
| `multi_assembly.py` | Fig. 6 B: multi-assembly model in its three regimes |
| `figures/` | Output of the two scripts (`.pdf` and transparent `.png`) |

## Running

```bash
pip install brian2 numpy matplotlib
python two_layer.py          # ~20 s -> figures/two_layer_raster.pdf/.png
python multi_assembly.py     # ~10 s -> figures/multi_assembly_raster.pdf/.png
```

Tested with Python 3.12, Brian2 2.9.0, NumPy 2.3 and Matplotlib 3.10.

Each model is a class.  Every parameter is a constructor argument, and
`run(duration, discard)` returns:
- the excitatory spike times and indices;
- the spike count of every layer or module in 1 ms bins;
- the mean excitatory firing rate.

```python
from models import TwoLayerModel, MultiAssemblyModel

res = MultiAssemblyModel(p_rewire=0.22, seed=1).run(6000, discard=1000)
res['counts'].shape          # (5000, 4): spikes per module per ms
res['rate']                  # mean excitatory firing rate (Hz)
```

## Models

Both models share these components:
- **Neurons:** Izhikevich neurons, regular-spiking excitatory and fast-spiking
  inhibitory.  Each neuron's parameters are spread by a random r ~ U(0, 1):
  - E: a = 0.02, b = 0.2, c = −65 + 15r², d = 8 − 6r²
  - I: a = 0.02 + 0.08r, b = 0.25 − 0.05r, c = −65, d = 2
- **Integration:** Euler.
- **Synapses:** spikes arriving during a millisecond are summed and delivered
  as a constant current over the next millisecond.

Each figure script lists the parameters it uses at the top.

### Two-layer model: `two_layer.py` (Fig. 4 B–D)

| | |
|---|---|
| Excitatory | 2 layers × 400 (superficial, deep) |
| Inhibitory | 200 |
| E→E | directed stochastic block model, no self-connections, weight 10; within-layer p = 0.15; inter-layer p = 0 or 0.025 |
| E→I, I→E, I→I | random, connection probability 0.25, weights +6, −6, −2 |
| Delays | E→E and E→I uniform on {1, …, 10} ms; inhibitory synapses 10 ms |
| Background | every ms, each neuron receives 15 with probability 0.3, plus 5·U(0, 1) |
| Integration | dt = 0.1 ms; a neuron that fires is held until the end of that millisecond, then reset |
| Run | 2500 ms, first 500 ms discarded, seed 1 |

The script draws three panels.  Each shows the population activity V(t), the
total excitatory spikes per ms, above a raster coloured by layer:

| Panel | Inter-layer connectivity | Behaviour |
|---|---|---|
| B | none | the layers oscillate in alternation, so V(t) has half the period of either layer |
| C | superficial → deep only (0.025) | metastable: the deep layer sometimes fires with the superficial one and sometimes on its own |
| D | both directions (0.025) | the layers synchronise into one slower oscillation |

### Multi-assembly model: `multi_assembly.py` (Fig. 6 B)

| | |
|---|---|
| Excitatory | 4 modules × 200 |
| Inhibitory | 200 |
| E→E | per module, 14 × 200 = 2800 directed edges drawn uniformly over distinct pairs (mean out-degree 14), weight 17 |
| Rewiring | each E→E edge, with probability p, keeps its source and gets a new target drawn uniformly from a uniformly chosen module |
| E→I | each inhibitory neuron receives from 4 consecutive excitatory neurons of one module, weight 50·U(0, 1) |
| I→E, I→I | all-to-all, weights −2·U(0, 1), −1·U(0, 1) |
| Delays | E→E uniform on {1, …, 20} ms; all others 1 ms |
| Background | each neuron receives 15 for 1 ms with probability 1 − e^−0.01 per ms |
| Integration | dt = 0.2 ms |
| Run | 6000 ms, first 1000 ms discarded, 2000–3000 ms shown, seed 1 |

The script draws three panels, one per rewiring probability.  Each shows V(t)
above a raster coloured by module:

| p | Regime | Behaviour |
|---|---|---|
| 0 | winner-take-all | modules activate in alternation |
| 0.22 | critical | irregular, scale-free activity |
| 0.6 | oscillatory | the modules synchronise into gamma bursts |
