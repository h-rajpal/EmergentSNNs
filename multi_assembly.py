"""Figure 6 B: the multi-assembly model in its three regimes.

    python multi_assembly.py     # -> figures/multi_assembly_raster.pdf / .png

For each rewiring probability, the population activity V(t) (total excitatory
spikes per 1 ms) above a raster of the excitatory neurons, coloured by module:

  p = 0     winner-take-all: modules activate in alternation;
  p = 0.22  critical: irregular, scale-free activity;
  p = 0.6   oscillatory: the modules synchronise into gamma bursts.

Operating point
---------------
Network
  excitatory modules     4 x 200 Izhikevich regular-spiking neurons
  inhibitory             200 Izhikevich fast-spiking neurons
  E -> E                 within each module 14 x 200 = 2800 directed edges
                         drawn uniformly over distinct pairs (mean out-degree
                         14), weight 17; each edge rewired with probability p
                         to a uniform neuron of a uniform module (any module,
                         own included), source kept, so out-degrees are
                         preserved
  E -> I                 focal: each I neuron receives from 4 consecutive E
                         neurons of one module, weight 50 * U(0, 1)
  I -> E, I -> I         all-to-all, weights -2 * U(0, 1) and -1 * U(0, 1)
  delays                 E->E uniform on {1, ..., 20} ms, all others 1 ms
Drive and integration
  background             each neuron receives 15 for 1 ms with probability
                         1 - exp(-0.01) per ms
  integration            Euler, dt = 0.2 ms; 1 ms synaptic current pulses
Run
  6000 ms simulated, the first 1000 ms discarded; the panel shows
  2000-3000 ms; seed 1.
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from models import MultiAssemblyModel

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- parameters
NETWORK = dict(n_modules=4, n_exc_per_module=200, n_inh=200,
               edges_per_node=14, rewire_any_module=True, w_ee=17.0,
               w_ei=50.0, w_ie=2.0, w_ii=1.0, d_max_ee=20, d_other=1,
               i_bg=15.0, lam_bg=0.01, dt=0.2)
REGIMES = [(0.0, 'WTA', '#7B6D8D'),
           (0.22, 'Critical', '#EB663B'),
           (0.6, 'Oscillatory', '#1F4E73')]
DURATION, DISCARD, SEED = 6000.0, 1000.0, 1
T0, WINDOW = 2000.0, 1000.0             # displayed window (ms, absolute)

# ---------------------------------------------------------------- style
MODULES = ['#E87040', '#93B83C', '#4FA3A5', '#9B6A9B']
NEUTRAL = '#333333'
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})


def main():
    npc = NETWORK['n_exc_per_module']
    n_exc = NETWORK['n_modules'] * npc
    colours = np.array(MODULES)
    fig = plt.figure(figsize=(3.4, 1.15 * len(REGIMES)))
    outer = fig.add_gridspec(len(REGIMES), 1, hspace=0.55)
    first = None
    for k, (p, label, colour) in enumerate(REGIMES):
        res = MultiAssemblyModel(p_rewire=p, seed=SEED, **NETWORK).run(
            DURATION, DISCARD)
        print(f'p = {p:<5g}{label:<12} rate {res["rate"]:5.2f} Hz')

        inner = outer[k].subgridspec(2, 1, height_ratios=[1, 2.6], hspace=0.06)
        top = fig.add_subplot(inner[0], sharex=first)
        first = first or top
        ax = fig.add_subplot(inner[1], sharex=first)

        # counts[0] is the millisecond starting at DISCARD.
        lo = int(round(T0 - DISCARD))
        V = res['counts'].sum(axis=1)[lo:lo + int(WINDOW)]
        top.fill_between(np.arange(V.size), V, color=NEUTRAL, lw=0,
                         step='mid')
        top.set_ylabel('$V(t)$', fontsize=7)
        top.set_ylim(0, max(V.max() * 1.05, 1))
        top.yaxis.set_major_locator(MaxNLocator(nbins=2, prune='lower'))
        top.tick_params(labelbottom=False)
        top.set_title(f'$p$ = {p:g} ({label})', fontsize=7.5, loc='left',
                      pad=2, color=colour)

        t, i = res['t'], res['i']
        m = (t >= T0) & (t < T0 + WINDOW)
        ax.scatter(t[m] - T0, i[m], s=0.15, c=colours[i[m] // npc],
                   linewidths=0, rasterized=True)
        ax.set_ylim(0, n_exc)
        ax.set_ylabel('Neuron', fontsize=7)
        for b in range(npc, n_exc, npc):
            ax.axhline(b, color='0.85', lw=0.4)
        if k < len(REGIMES) - 1:
            ax.tick_params(labelbottom=False)
    ax.set_xlabel('Time (ms)')
    ax.set_xlim(0, WINDOW)

    # One panel letter, level with the first title, left of every y-label.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [inv.transform_bbox(a.get_tightbbox(r)) for a in fig.axes]
    fig.text(min(b.x0 for b in boxes), boxes[0].y1, 'B', fontsize=11,
             fontweight='bold', ha='left', va='top')

    out = os.path.join(HERE, 'figures', 'multi_assembly_raster')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + '.pdf', dpi=400, bbox_inches='tight')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight', transparent=True)
    print('wrote', out + '.pdf / .png')


if __name__ == '__main__':
    main()
