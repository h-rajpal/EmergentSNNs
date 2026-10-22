"""Brian2 implementations of the two spiking-network models of the paper.

    TwoLayerModel        two excitatory layers (superficial, deep) and a common
                         inhibitory population in a PING architecture (Fig. 4).
    MultiAssemblyModel   four excitatory modules and a common, densely connected
                         inhibitory population, with a rewiring probability p
                         between modules (Fig. 6).

Both use Izhikevich neurons (regular-spiking excitatory, fast-spiking
inhibitory, with the usual small spread of parameters), Euler integration, a
Poisson-like background drive, and synaptic input delivered as a rectangular
1 ms current pulse: spikes arriving during one millisecond are summed and held
constant over the next.  ``run()`` returns the excitatory spikes and the spike
count of every layer/module in 1 ms bins.
"""

import glob
import os
import sys

import numpy as np


def _configure_compiler():
    """Use the conda-forge clang toolchain if present (Brian2's Cython target).

    Some conda environments ship a clang++ that cannot link; Brian2 then falls
    back silently to its ~5x slower NumPy target.  Does nothing if CXX is set.
    """
    if os.environ.get('CXX'):
        return
    for cxx in sorted(glob.glob(os.path.join(sys.prefix, 'bin', '*-clang++'))):
        if os.path.exists(cxx[:-2]):
            os.environ['CXX'], os.environ['CC'] = cxx, cxx[:-2]
            return


_configure_compiler()

import brian2 as b2                                            # noqa: E402
from brian2 import ms                                          # noqa: E402

b2.BrianLogger.suppress_name('resolution_conflict', 'device')
b2.prefs.logging.console_log_level = 'ERROR'

# Izhikevich membrane equations; I_syn is the 1 ms synaptic pulse, I_stage
# collects the arrivals for the next millisecond, I_bg is the background drive.
# ``frozen`` (two-layer model only) marks a neuron that fired this millisecond.
_EQS = '''
dv/dt = {gate}(0.04*v*v + 5*v + 140 - u + I_tot) / ms : 1
du/dt = {gate}(a*(b*v - u)) / ms : 1
I_tot = I_bg + I_syn : 1
I_bg : 1
I_syn : 1
I_stage : 1
a : 1 (constant)
b : 1 (constant)
c : 1 (constant)
d : 1 (constant)
{extra}'''


def _izhikevich_parameters(G, n_exc, n_inh, rng):
    """Regular-spiking E and fast-spiking I parameters (Izhikevich 2003)."""
    r = rng.random(n_exc)
    G.a[:n_exc], G.b[:n_exc] = 0.02, 0.2
    G.c[:n_exc] = -65 + 15 * r ** 2
    G.d[:n_exc] = 8 - 6 * r ** 2
    r = rng.random(n_inh)
    G.a[n_exc:] = 0.02 + 0.08 * r
    G.b[n_exc:] = 0.25 - 0.05 * r
    G.c[n_exc:], G.d[n_exc:] = -65.0, 2.0


def _connect(G, name, src, tgt, w, delay):
    S = b2.Synapses(G, G, 'w : 1', on_pre='I_stage_post += w', name=name)
    S.connect(i=np.asarray(src, int), j=np.asarray(tgt, int))
    S.w = w
    S.delay = delay
    return S


def _record(net, G, n_exc, sizes, dt, duration, discard):
    """Run, and bin the excitatory spikes into 1 ms counts per group."""
    mon = b2.SpikeMonitor(G[:n_exc], name='spikes')
    net.add(mon)
    net.run(duration * ms, report=None)
    t, i = np.asarray(mon.t / ms), np.asarray(mon.i)
    # Bin by integer time step, not floor(t): a spike at exactly 6 ms can come
    # back as 5.999... ms.
    t_ms = np.rint(t / dt).astype(np.int64) // int(round(1 / dt))
    keep = t_ms >= discard
    t, i, t_ms = t[keep], i[keep], t_ms[keep]
    n_bins = int(round(duration - discard))
    group = np.searchsorted(np.cumsum(sizes), i, side='right')
    counts = np.zeros((n_bins, len(sizes)), dtype=np.int32)
    np.add.at(counts, (t_ms - int(round(discard)), group), 1)
    return {'t': t, 'i': i, 'counts': counts,
            'rate': counts.sum() / (n_bins * 1e-3) / n_exc}


class TwoLayerModel:
    """Two-layer PING model (Fig. 4).

    Two excitatory layers, ordered (superficial, deep), and one inhibitory
    population.

    Parameters
    ----------
    n_exc : neurons per excitatory layer, (superficial, deep).
    n_inh : inhibitory neurons.
    p_ee : 2x2 E->E connection probabilities, p_ee[i][j] from layer i to
        layer j (stochastic block model, directed, no self-loops).  The
        diagonal holds the within-layer probabilities, p_ee[0][1] is
        superficial -> deep (p_cross in the paper) and p_ee[1][0] deep ->
        superficial.
    d_max : E->E and E->I delays are uniform on {1, ..., d_max} ms; every
        inhibitory synapse has delay d_max.
    w_ee : E->E weight (binary connectivity).
    w_ei, w_ie, w_ii : E->I, I->E and I->I weights, each on a random mask of
        density ``p_inh`` (I->E and I->I inhibitory, i.e. negative).
    i_peak, p_peak, i_base : background drive; every millisecond each neuron
        receives i_peak with probability p_peak plus i_base * U(0, 1).
    dt : Euler step (ms).
    """

    def __init__(self, n_exc=(400, 400), n_inh=200,
                 p_ee=((0.15, 0.0), (0.0, 0.15)), d_max=10, w_ee=10.0,
                 w_ei=6.0, w_ie=6.0, w_ii=2.0, p_inh=0.25, i_peak=15.0,
                 p_peak=0.3, i_base=5.0, dt=0.1, seed=None):
        self.n_exc, self.n_inh = tuple(n_exc), n_inh
        self.p_ee = np.asarray(p_ee, float)
        self.d_max, self.dt, self.seed = d_max, dt, seed
        self.w_ee, self.w_ei, self.w_ie, self.w_ii = w_ee, w_ei, w_ie, w_ii
        self.p_inh = p_inh
        self.i_peak, self.p_peak, self.i_base = i_peak, p_peak, i_base

    def connectivity(self, rng):
        """(name, src, tgt, weight, delay in ms) for every pathway."""
        ne = sum(self.n_exc)
        n = ne + self.n_inh
        edge = np.cumsum([0, *self.n_exc])
        src, tgt = [], []
        for a in range(2):
            for b in range(2):
                s, t = np.nonzero(rng.random((self.n_exc[a], self.n_exc[b]))
                                  < self.p_ee[a, b])
                src.append(s + edge[a])
                tgt.append(t + edge[b])
        src, tgt = np.concatenate(src), np.concatenate(tgt)
        ok = src != tgt
        out = [('S_ee', src[ok], tgt[ok], self.w_ee,
                rng.integers(1, self.d_max + 1, ok.sum()))]
        for name, (s0, s1), (t0, t1), w, spread in (
                ('S_ei', (0, ne), (ne, n), self.w_ei, True),
                ('S_ie', (ne, n), (0, ne), -self.w_ie, False),
                ('S_ii', (ne, n), (ne, n), -self.w_ii, False)):
            s, t = np.nonzero(rng.random((s1 - s0, t1 - t0)) < self.p_inh)
            d = (rng.integers(1, self.d_max + 1, s.size) if spread
                 else np.full(s.size, self.d_max))
            out.append((name, s + s0, t + t0, w, d))
        return out

    def _drive(self):
        return 'I_bg = i_peak * int(rand() < p_peak)\nI_bg += i_base * rand()'

    def build(self):
        rng = np.random.default_rng(self.seed)
        if self.seed is not None:
            b2.seed(self.seed)
        b2.defaultclock.dt = self.dt * ms
        ne = sum(self.n_exc)
        # A neuron that crosses v > 30 is frozen for the rest of that
        # millisecond and reset (v = c, u += d) at the start of the next one,
        # so it fires at most once per millisecond.
        # (Brian2's own refractoriness cannot be used: it also blocks the tick
        # below from resetting v and u of refractory neurons.)
        G = b2.NeuronGroup(ne + self.n_inh,
                           _EQS.format(gate='(1 - frozen) * ',
                                       extra='frozen : 1\n'),
                           threshold='v > 30 and frozen == 0',
                           reset='frozen = 1', method='euler', name='neurons',
                           namespace={'i_peak': self.i_peak, 'p_peak': self.p_peak,
                                      'i_base': self.i_base})
        _izhikevich_parameters(G, ne, self.n_inh, rng)
        G.v, G.u = -65.0, -1.0
        G.run_regularly('u += d * frozen\n'
                        'v = c * frozen + v * (1 - frozen)\n'
                        'frozen = 0\n'
                        'I_syn = I_stage\n'
                        'I_stage = 0\n' + self._drive(),
                        dt=1 * ms, when='start', name='ms_tick')
        # A spike in millisecond t with delay D acts during millisecond t + D.
        # The tick above swaps staged input in at the start of
        # each millisecond, so the Brian2 delay is D - 1.
        syn = [_connect(G, name, s, t, w, (d - 1) * ms)
               for name, s, t, w, d in self.connectivity(rng)]
        return b2.Network(G, *syn), G

    def run(self, duration, discard=0.0):
        """Simulate ``duration`` ms and keep everything from ``discard`` on.

        Returns a dict with the excitatory spike times ``t`` (ms) and indices
        ``i``, ``counts`` (n_ms, 2) spikes per layer per ms, and the mean
        excitatory ``rate`` (Hz).
        """
        b2.start_scope()
        net, G = self.build()
        return _record(net, G, sum(self.n_exc), self.n_exc, self.dt,
                       duration, discard)


class MultiAssemblyModel:
    """Multi-assembly model (Fig. 6).

    ``n_modules`` excitatory modules and one inhibitory population.  Within each
    module, ``edges_per_node * n_exc_per_module`` directed E->E edges are drawn
    uniformly over distinct pairs.  Each edge is then rewired with probability
    ``p_rewire``: its source is kept, so every neuron's out-degree is preserved,
    and its target becomes a uniform neuron of a uniformly drawn module
    (``rewire_any_module=True``: all modules including the source's own, as
    used for the paper's results; False: one of the other modules only).
    Self-loops and duplicates are redrawn.

    Inhibition: each inhibitory neuron receives from a focal block of
    n_exc / n_inh consecutive excitatory neurons (within one module) and
    projects to every excitatory and every inhibitory neuron.

    Parameters follow the original Octave/MATLAB code of Shanahan (2010) and
    Mediano & Shanahan (2017), with out-degree 14 and d_max_ee 20 ms for the
    paper.  Weights: E->E constant w_ee; E->I, I->E, I->I are factor * U(0, 1)
    (inhibitory ones negative).  Delays: E->E uniform on {1, ..., d_max_ee} ms,
    every other synapse d_other ms.  Background: each neuron receives i_bg for
    1 ms with probability 1 - exp(-lam_bg) per ms.
    """

    def __init__(self, p_rewire=0.0, n_modules=4, n_exc_per_module=200,
                 n_inh=200, edges_per_node=14, rewire_any_module=True,
                 w_ee=17.0, w_ei=50.0, w_ie=2.0, w_ii=1.0, d_max_ee=20,
                 d_other=1, i_bg=15.0, lam_bg=0.01, dt=0.2, seed=None):
        self.p_rewire, self.n_modules = p_rewire, n_modules
        self.n_exc_per_module, self.n_inh = n_exc_per_module, n_inh
        self.edges_per_node = edges_per_node
        self.rewire_any_module = rewire_any_module
        self.w_ee, self.w_ei, self.w_ie, self.w_ii = w_ee, w_ei, w_ie, w_ii
        self.d_max_ee, self.d_other = d_max_ee, d_other
        self.i_bg, self.lam_bg, self.dt, self.seed = i_bg, lam_bg, dt, seed

    @property
    def n_exc(self):
        return self.n_modules * self.n_exc_per_module

    def topology(self, rng):
        """Directed E->E edges (source, target) after rewiring."""
        npc, nm = self.n_exc_per_module, self.n_modules
        n_edges = self.edges_per_node * npc
        src = np.empty(nm * n_edges, dtype=np.int64)
        tgt = np.empty_like(src)
        for m in range(nm):
            # Distinct directed pairs without self-loops: k -> (i, j != i).
            k = rng.choice(npc * (npc - 1), size=n_edges, replace=False)
            i, j = k // (npc - 1), k % (npc - 1)
            sl = slice(m * n_edges, (m + 1) * n_edges)
            src[sl], tgt[sl] = m * npc + i, m * npc + j + (j >= i)
        if self.p_rewire > 0:
            edges = set(zip(src.tolist(), tgt.tolist()))
            for e in np.flatnonzero(rng.random(src.size) < self.p_rewire):
                s, t_old = int(src[e]), int(tgt[e])
                edges.discard((s, t_old))
                for _ in range(100):           # redraw self-loops / duplicates
                    if self.rewire_any_module:
                        mod = rng.integers(nm)
                    else:
                        mod = (s // npc + 1 + rng.integers(nm - 1)) % nm
                    t_new = int(mod * npc + rng.integers(npc))
                    if t_new != s and (s, t_new) not in edges:
                        break
                else:
                    t_new = t_old
                edges.add((s, t_new))
                tgt[e] = t_new
        return src, tgt

    def build(self):
        rng = np.random.default_rng(self.seed)
        if self.seed is not None:
            b2.seed(self.seed)
        b2.defaultclock.dt = self.dt * ms
        ne, ni = self.n_exc, self.n_inh
        G = b2.NeuronGroup(ne + ni, _EQS.format(gate='', extra=''),
                           threshold='v >= 30', reset='v = c; u += d',
                           method='euler', name='neurons',
                           namespace={'i_bg': self.i_bg,
                                      'p_spike': 1.0 - np.exp(-self.lam_bg)})
        _izhikevich_parameters(G, ne, ni, rng)
        G.v = -65
        G.u = 'b * v'
        G.run_regularly('I_syn = I_stage\nI_stage = 0\n'
                        'I_bg = i_bg * int(rand() < p_spike)',
                        dt=1 * ms, when='start', name='ms_tick')

        src, tgt = self.topology(rng)
        syn = [_connect(G, 'S_ee', src, tgt, self.w_ee,
                        rng.integers(1, self.d_max_ee + 1, src.size) * ms)]
        # Focal E->I: consecutive blocks of n_exc / n_inh E neurons per I.
        e = np.arange(ne)
        syn.append(_connect(G, 'S_ei', e, ne + e // (ne // ni),
                            rng.random(ne) * self.w_ei, self.d_other * ms))
        # Diffuse I->E and I->I: all-to-all.
        for name, lo, hi, factor in (('S_ie', 0, ne, self.w_ie),
                                     ('S_ii', ne, ne + ni, self.w_ii)):
            pre = np.repeat(np.arange(ne, ne + ni), hi - lo)
            post = np.tile(np.arange(lo, hi), ni)
            syn.append(_connect(G, name, pre, post,
                                -rng.random(pre.size) * factor,
                                self.d_other * ms))
        return b2.Network(G, *syn), G

    def run(self, duration, discard=0.0):
        """Simulate ``duration`` ms and keep everything from ``discard`` on.

        Returns a dict with the excitatory spike times ``t`` (ms) and indices
        ``i``, ``counts`` (n_ms, n_modules) spikes per module per ms, and the
        mean excitatory ``rate`` (Hz).
        """
        b2.start_scope()
        net, G = self.build()
        return _record(net, G, self.n_exc,
                       [self.n_exc_per_module] * self.n_modules, self.dt,
                       duration, discard)
