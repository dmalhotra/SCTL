#!/usr/bin/env python3
"""Tables and plot data of the Numerical results section, from bench-cubed-sphere outputs in data/.

Convergence: data/{rome-128,genoa-96,icelake-64}.txt, the tolerance sweep at 12 patches per face.
OpenMP scaling: data/scaling-mpi-tol1e-3-genoa-96-7204544.txt, the one-process thread sweep of the new code.
Keeps the Duffy rows, writes data/scaling-genoa-<kernel>.dat for the plots, prints both RST tables and the
quantities the text quotes, and with an RST path as argument replaces the two tables in that file.
"""
import re, statistics, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
MACHINES = [('Rome', 'rome-128', 128), ('Genoa', 'genoa-96', 96), ('Icelake', 'icelake-64', 64)]
SCALING_FILE = 'scaling-mpi-tol1e-3-genoa-96-7204544.txt'
SCALING_CORES = 96
SCHEME = 'Duffy'
N_SCALING = 6 * 8 * 8 * 144  # nodes at order 12, 8 patches per face
TWIST = {'0.5236': r'\pi/6', '1.5708': r'\pi/2', '3.1416': r'\pi'}


def read(fname):
    """Rows of one output file as dicts keyed by the header names, plus the '== ...' section each row is in."""
    rows, names, section = [], None, ''
    for line in (HERE / 'data' / fname).read_text().splitlines():
        if line.startswith('== '):
            section = line[3:]
        elif line.startswith('#kernel'):
            names = [t for t in line[1:].split() if t != '|']
        elif names and re.match(r'\s*(laplace|stokes)\s', line):
            r = dict(zip(names, [t for t in line.split() if t != '|']))
            r['section'] = section
            rows.append(r)
    return rows


def grid(widths, header_rows, body_groups, hspan_top, caption):
    """An RST grid table; the first cell of each body group spans the group's rows, hspan_top gives the
    row-1 header cells spanning several columns."""
    ncol = len(widths)

    def line(cols, ch='-'):
        s = ''
        for c in range(ncol):
            s += '+' if (c in cols or (c - 1) in cols) else '|'
            s += ch * (widths[c] + 2) if c in cols else ' ' * (widths[c] + 2)
        return s + ('+' if (ncol - 1) in cols else '|')

    def row(cells, spans={}):
        s, c = '|', 0
        while c < ncol:
            n = spans.get(c, 1)
            s += ' ' + (cells[c] or '').ljust(sum(widths[c:c + n]) + 3 * (n - 1)) + ' |'
            c += n
        return s

    out = [line(set(range(ncol))), row(header_rows[0], hspan_top)]
    out.append(line({c for c0, n in hspan_top.items() for c in range(c0, c0 + n)}))
    out += [row(header_rows[1]), line(set(range(ncol)), '=')]
    for g in body_groups:
        for i, r in enumerate(g):
            out.append(row([None if (i and k == 0) else v for k, v in enumerate(r)]))
            out.append(line(set(range(ncol)) if i == len(g) - 1 else set(range(1, ncol))))
    body = '\n'.join('   ' + l for l in out)
    return '.. table:: ' + caption + '\n\n' + body + '\n'


def sci(s):
    """'7.17e-14' as :math:`7.17\\times10^{-14}`."""
    m, e = ('%.1e' % float(s)).split('e')
    return r':math:`%s\times10^{%d}`' % (m, int(e))


# convergence: Duffy rows of the tolerance sweep on each machine
conv = {name: {(r['kernel'], r['twist'], r['tol']): r for r in read(f + '.txt') if r['scheme'] == SCHEME and r['ppf'] == '12'}
        for name, f, _ in MACHINES}

# cross-machine agreement of the error columns: error-density and the tight-tolerance errors agree; at the two
# loose tolerances the AVX-512 machines take one Newton iteration fewer in approx_rsqrt and their errors are larger
worst_den, worst_tight, worst_loose = 0, 0, 1
for key, r0 in conv['Rome'].items():
    for name in ('Genoa', 'Icelake'):
        r = conv[name][key]
        worst_den = max(worst_den, abs(float(r['greens_den']) / float(r0['greens_den']) - 1))
        ratio = float(r['greens_sol']) / float(r0['greens_sol'])
        if key[2] in ('1e-09', '1e-12'):
            worst_tight = max(worst_tight, abs(ratio - 1))
        else:
            worst_loose = max(worst_loose, ratio)
print('Genoa/Icelake vs Rome: error-density within %.1f%%; error within %.1f%% at 1e-9 and 1e-12, up to %.0fx larger at 1e-3 and 1e-6'
      % (100 * worst_den, 100 * worst_tight, worst_loose))

groups = []
for kernel in ('laplace',):
    for tw in ('0.5236', '1.5708', '3.1416'):
        g = []
        for tol in ('1e-03', '1e-06', '1e-09', '1e-12'):
            r = conv['Rome'][(kernel, tw, tol)]
            pps = [conv[name][(kernel, tw, tol)]['pps/c_sl'] for name, _, _ in MACHINES]
            g.append([f'{kernel}, :math:`{TWIST[tw]}`', ':math:`10^{%d}`' % int(tol[2:]), sci(r['greens_den']),
                      sci(r['greens_sol'])] + ['%.0f' % float(p) for p in pps])
        groups.append(g)
conv_tab = grid(
    [22, 16, 27, 26, 8, 8, 9],
    [['kernel, twist', 'tol', 'error-density', 'error', 'pts/s/core, SL setup', None, None],
     [None, None, None, None, 'Rome', 'Genoa', 'Icelake']], groups, {4: 3},
    """Order 12, 12 patches per face (864 elements, 124,416 nodes), Laplace, single point source
   outside the surface. *error* is :math:`\\max|(S[\\partial_n u]-D[u])-u|/\\max|u|` at the surface nodes;
   *error-density* is the interpolation error of the densities at off-node parameters. Errors from
   Rome; throughput per core with every core of the machine in use.""")

# accuracy against throughput on Genoa, for the plot: one row per tolerance, a (pps, err) pair per kernel and twist
with open(HERE / 'data' / 'convergence-genoa.dat', 'w') as out:
    cols = [(k, tw) for k in ('laplace', 'stokes') for tw in ('0.5236', '1.5708', '3.1416')]
    out.write('tol ' + ' '.join(f'{k[:3]}{i}_pps {k[:3]}{i}_err' for k, tw in cols for i in [TWIST[tw].strip(chr(92)).replace("/", "")]) + '\n')
    for tol in ('1e-03', '1e-06', '1e-09', '1e-12'):
        out.write(tol + ' ' + ' '.join('%s %s' % (conv['Genoa'][(k, tw, tol)]['pps/c_sl'], conv['Genoa'][(k, tw, tol)]['greens_sol']) for k, tw in cols) + '\n')

# OpenMP scaling: the one-process thread sweep of the new code; the full-node point also appears in the MPI
# part of the file, so each thread count takes the smaller of its setup times
scal = {}
for r in read(SCALING_FILE):
    m = re.match(r'new (laplace|stokes) processes 1 threads (\d+)$', r['section'])
    if m and r['scheme'] == SCHEME:
        key = (m.group(1), int(m.group(2)))
        scal[key] = min(scal.get(key, float('inf')), float(r['setup_sl']))
rows = []
for kernel in ('laplace', 'stokes'):
    thr = sorted(t for k, t in scal if k == kernel)
    assert thr[0] == 1 and thr[-1] == SCALING_CORES, thr
    t1, tn = scal[(kernel, 1)], scal[(kernel, SCALING_CORES)]
    with open(HERE / 'data' / f'scaling-genoa-{kernel}.dat', 'w') as out:
        out.write('thr setup speedup eff pps\n')  # pps: points per second per core
        for t in thr:
            s = scal[(kernel, t)]
            out.write('%d %.4f %.4f %.2f %.1f\n' % (t, s, t1 / s, 100 * t1 / s / t, N_SCALING / s / t))
    rows.append([kernel.capitalize(), '%.3f' % t1, '{:,.0f}'.format(N_SCALING / t1), '%.3f' % tn,
                 '{:,.0f}'.format(N_SCALING / tn), '%.1f× (%.0f%%)' % (t1 / tn, 100 * t1 / tn / SCALING_CORES)])
scal_tab = grid(
    [9, 11, 9, 11, 11, 13],
    [['kernel', '1 thread', None, '96 threads', None, 'speedup'],
     [None, 'setup (s)', 'pts/s', 'setup (s)', 'pts/s', None]], [[r] for r in rows], {1: 2, 3: 2},
    """*setup* is the wall time of the single-layer self- and near-interaction setup; *speedup* is
   against one thread, with the parallel efficiency in parentheses.""")

if len(sys.argv) > 1:
    p = pathlib.Path(sys.argv[1])
    rst = p.read_text()
    for start, end, new in [('.. table:: Order 12, 12 patches per face', 'Genoa and Icelake match', conv_tab),
                            ('.. table:: *setup* is the total wall time', '- **', scal_tab)]:
        a = rst.index(start)
        b = rst.index(end, a)
        rst = rst[:a] + new + '\n' + rst[b:]
    p.write_text(rst)
    print('tables replaced in', p)
else:
    print(conv_tab)
    print(scal_tab)

# quantities the text quotes
R = conv['Rome']
f = lambda name, k, tw, tol, col='pps/c_sl': float(conv[name][(k, tw, tol)][col])
print('\n== convergence quantities (Duffy)')
for tw in TWIST:
    print('Genoa laplace %-6s pps/c by tol: %s' % (TWIST[tw], ' '.join('%.0f' % f('Genoa', 'laplace', tw, tol) for tol in ('1e-03', '1e-06', '1e-09', '1e-12'))))
print('laplace pi/6 error: 1e-3 %s -> 1e-12 %s' % (R[('laplace', '0.5236', '1e-03')]['greens_sol'], R[('laplace', '0.5236', '1e-12')]['greens_sol']))
print('laplace 1e-12 error: pi %s vs pi/6 %s, ratio %.0f' % (R[('laplace', '3.1416', '1e-12')]['greens_sol'], R[('laplace', '0.5236', '1e-12')]['greens_sol'],
      float(R[('laplace', '3.1416', '1e-12')]['greens_sol']) / float(R[('laplace', '0.5236', '1e-12')]['greens_sol'])))
print('tolerance cost, laplace pi/6 Rome pps/c: 1e-3 %.0f, 1e-12 %.0f, ratio %.2f' % (f('Rome', 'laplace', '0.5236', '1e-03'), f('Rome', 'laplace', '0.5236', '1e-12'),
      f('Rome', 'laplace', '0.5236', '1e-03') / f('Rome', 'laplace', '0.5236', '1e-12')))
for name in ('Rome', 'Genoa', 'Icelake'):
    print('twist cost at 1e-9, laplace %s: pi/6 %.0f -> pi %.0f, ratio %.2f' % (name, f(name, 'laplace', '0.5236', '1e-09'), f(name, 'laplace', '3.1416', '1e-09'),
          f(name, 'laplace', '0.5236', '1e-09') / f(name, 'laplace', '3.1416', '1e-09')))
for name in ('Genoa', 'Icelake'):
    rr = [f(name, k, tw, tol) / f('Rome', k, tw, tol) for k in ('laplace', 'stokes') for tw in TWIST for tol in ('1e-03', '1e-06', '1e-09', '1e-12')]
    print('  %s/Rome pps/c over %d comparisons: median %.2f, min %.2f, max %.2f' % (name, len(rr), statistics.median(rr), min(rr), max(rr)))
print('\n== OpenMP scaling quantities (Duffy, tol 1e-3, Genoa)')
for kernel in ('laplace', 'stokes'):
    t = {thr: s for (k, thr), s in scal.items() if k == kernel}
    pc = lambda thr: N_SCALING / t[thr] / thr
    e = lambda thr: 100 * t[1] / t[thr] / thr
    print('%-8s per-core 1thr %.0f -> 96thr %.0f (%+.0f%%); efficiency ' % (kernel, pc(1), pc(96), 100 * (pc(96) / pc(1) - 1))
          + ' '.join('%d:%.0f%%' % (thr, e(thr)) for thr in sorted(t)) + '; full node %.0f pts/s, setup %.3f s' % (N_SCALING / t[96], t[96]))
