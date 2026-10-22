"""Figure 4 B-D: the two-layer PING model for three inter-layer connectivities.

    python two_layer.py          # -> figures/two_layer_raster.pdf / .png

For each connectivity, the population activity V(t) (total excitatory spikes
per 1 ms) above a raster of the excitatory neurons, coloured by layer:

  B  no inter-layer connections: the layers oscillate in alternation, and V(t)
     has half the period of either layer;
  C  asymmetric (superficial -> deep only): metastable, the deep layer
     sometimes fires with the superficial one and sometimes on its own;
  D  bi-directional: the layers synchronise into one slower oscillation, and
     V(t) has the period of the layers.

Operating point
---------------
Network
  excitatory layers      2 x 400 Izhikevich regular-spiking neurons,
                         order (superficial, deep); plotted deep at the bottom
  inhibitory             200 Izhikevich fast-spiking neurons
  E -> E                 directed stochastic block model, weight 10
                           within-layer probability       0.15 (both layers)
                           inter-layer probability        0 or 0.025, below
  E -> I, I -> E, I -> I random masks of density 0.25, weights +6, -6, -2
  delays                 E->E and E->I uniform on {1, ..., 10} ms;
                         every inhibitory synapse 10 ms
Drive and integration
  background             every ms each neuron receives 15 with probability
                         0.3, plus 5 * U(0, 1)
  integration            Euler, dt = 0.1 ms; 1 ms synaptic current pulses
Run
  2500 ms simulated, the first 500 ms discarded, 2000 ms shown; seed 1
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from models import TwoLayerModel

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- parameters
NETWORK = dict(n_exc=(400, 400), n_inh=200, d_max=10, w_ee=10.0, w_ei=6.0,
               w_ie=6.0, w_ii=2.0, p_inh=0.25, i_peak=15.0, p_peak=0.3,
               i_base=5.0, dt=0.1)
P_WITHIN = 0.15
P_CROSS = 0.025
# p_ee[i][j] is the probability of an edge from layer i to layer j.
CASES = [('B', 'No inter-layer interaction',
          ((P_WITHIN, 0.0), (0.0, P_WITHIN))),
         ('C', 'Asymmetric interaction',
          ((P_WITHIN, P_CROSS), (0.0, P_WITHIN))),
         ('D', 'Bi-directional interaction',
          ((P_WITHIN, P_CROSS), (P_CROSS, P_WITHIN)))]
DURATION, DISCARD, SEED = 2500.0, 500.0, 1

# ---------------------------------------------------------------- style
LAYERS = [('Superficial', '#E87040'), ('Deep', '#93B83C')]
NEUTRAL = '#333333'
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})


def main():
    n1, n2 = NETWORK['n_exc']
    fig = plt.figure(figsize=(3.4, 1.15 * len(CASES)))
    outer = fig.add_gridspec(len(CASES), 1, hspace=0.55)
    first, tops = None, []
    for k, (letter, title, p_ee) in enumerate(CASES):
        res = TwoLayerModel(p_ee=p_ee, seed=SEED, **NETWORK).run(DURATION,
                                                                DISCARD)
        corr = np.corrcoef(res['counts'].T)[0, 1]
        print(f'{letter} {title:<28} rate {res["rate"]:6.1f} Hz, '
              f'layer correlation {corr:+.2f}')

        inner = outer[k].subgridspec(2, 1, height_ratios=[1, 2.6], hspace=0.06)
        top = fig.add_subplot(inner[0], sharex=first)
        first = first or top
        ax = fig.add_subplot(inner[1], sharex=first)
        tops.append((letter, top))

        V = res['counts'].sum(axis=1)
        top.fill_between(np.arange(V.size), V, color=NEUTRAL, lw=0,
                         step='mid')
        top.set_ylabel('$V(t)$', fontsize=7)
        top.set_ylim(0, max(V.max() * 1.05, 1))
        top.yaxis.set_major_locator(MaxNLocator(nbins=2, prune='lower'))
        top.tick_params(labelbottom=False)
        top.set_title(title, fontsize=7.5, loc='left', pad=2, color=NEUTRAL)

        # Deep layer at the bottom, superficial on top, as in Fig. 4.
        t, i = res['t'] - DISCARD, res['i']
        sup = i < n1
        y = np.where(sup, i + n2, i - n1)
        colours = np.where(sup, LAYERS[0][1], LAYERS[1][1])
        ax.scatter(t, y, s=0.15, c=colours, linewidths=0, rasterized=True)
        ax.set_ylim(0, n1 + n2)
        ax.set_yticks([0, n2, n1 + n2])
        ax.set_ylabel('Neuron', fontsize=7)
        ax.axhline(n2, color='0.85', lw=0.4)
        if k < len(CASES) - 1:
            ax.tick_params(labelbottom=False)
    ax.set_xlabel('Time (ms)')
    ax.set_xlim(0, DURATION - DISCARD)

    # Panel letters on each title's baseline, left of every y-label.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    x0 = min(inv.transform_bbox(a.get_tightbbox(r)).x0 for a in fig.axes)
    for letter, top in tops:
        y0 = inv.transform_bbox(top.title.get_window_extent(r)).y0
        fig.text(x0, y0, letter, fontsize=11, fontweight='bold', ha='left',
                 va='bottom')

    out = os.path.join(HERE, 'figures', 'two_layer_raster')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + '.pdf', dpi=400, bbox_inches='tight')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight', transparent=True)
    print('wrote', out + '.pdf / .png')


if __name__ == '__main__':
    main()
