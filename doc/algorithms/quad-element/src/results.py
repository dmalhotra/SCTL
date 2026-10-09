#!/usr/bin/env python3
"""Tables and plot data of the Numerical results section, from the bench-cubed-sphere outputs in data/.

Reads data/{rome-128,genoa-96,icelake-64}.txt, keeps the Duffy rows, writes data/scaling-<machine>-<kernel>.dat
for the plots, prints both RST tables and the quantities the text quotes, and with an RST path as argument
replaces the two tables in that file.
"""
import re, statistics, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
MACHINES = [('Rome', 'rome-128', 128), ('Genoa', 'genoa-96', 96), ('Icelake', 'icelake-64', 64)]
SCHEME = 'Duffy'
N_SCALING = 6 * 8 * 8 * 144  # nodes at order 12, 8 patches per face
TWIST = {'0.5236': r'\pi/6', '1.5708': r'\pi/2', '3.1416': r'\pi'}


def read(fname):
    """Rows of one output file as dicts keyed by the header names."""
    rows, names = [], None
    for line in (HERE / 'data' / fname).read_text().splitlines():
        if line.startswith('#kernel'):
            names = [t for t in line[1:].split() if t != '|']
        elif names and re.match(r'\s*(laplace|stokes)\s', line):
            vals = [t for t in line.split() if t != '|']
            rows.append(dict(zip(names, vals)))
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
    m, e = s.split('e')
    return r':math:`%s\times10^{%d}`' % (m, int(e))


data = {name: [r for r in read(f + '.txt') if r['scheme'] == SCHEME] for name, f, _ in MACHINES}
conv = {name: {(r['kernel'], r['twist'], r['tol']): r for r in rows if r['ppf'] == '12'} for name, rows in data.items()}
scal = {name: {k: sorted((r for r in rows if r['ppf'] == '8' and r['tol'] == '1e-09' and r['kernel'] == k),
                         key=lambda r: int(r['thr'])) for k in ('laplace', 'stokes')} for name, rows in data.items()}

# cross-machine agreement of the error columns: error-density and the tight-tolerance errors agree; at the two
# loose tolerances the Genoa and Icelake runs give larger errors than Rome
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

# convergence table
groups = []
for kernel in ('laplace', 'stokes'):
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
    """Order 12, 12 patches per face (864 elements, 124,416 nodes), single point source
   outside the surface. *error* is :math:`\\max|(S[\\partial_n u]-D[u])-u|/\\max|u|` at the surface
   nodes, so it exercises both operators together. The two error columns are from Rome. Genoa and
   Icelake agree with them to within %d%% in error-density and, at tolerances :math:`10^{-9}` and
   :math:`10^{-12}`, in error; at :math:`10^{-3}` and :math:`10^{-6}` their errors are larger, by up to
   %d×. Throughput is the single-layer setup with each machine on all of its cores, so it carries that
   machine's parallel efficiency.""" % (-(-100 * max(worst_den, worst_tight) // 1), -(-worst_loose // 1)))

# scaling: .dat files and table
rows = []
for name, f, cores in MACHINES:
    for kernel in ('laplace', 'stokes'):
        s = scal[name][kernel]
        assert int(s[0]['thr']) == 1 and int(s[-1]['thr']) == cores, (name, kernel, [r['thr'] for r in s])
        t1 = float(s[0]['setup_sl'])
        with open(HERE / 'data' / f'scaling-{f.split("-")[0]}-{kernel}.dat', 'w') as out:
            out.write('thr setup speedup eff pps\n')  # pps: points per second per core
            for r in s:
                thr, t = int(r['thr']), float(r['setup_sl'])
                out.write('%d %.4f %.4f %.2f %.1f\n' % (thr, t, t1 / t, 100 * t1 / t / thr, N_SCALING / t / thr))
        tn = float(s[-1]['setup_sl'])
        rows.append([name, kernel.capitalize(), str(cores), '%.3f' % t1, '%.0f' % (N_SCALING / t1), '%.3f' % tn,
                     '{:,.0f}'.format(N_SCALING / tn), '%.1f× (%.0f%%)' % (t1 / tn, 100 * t1 / tn / cores)])
rows = rows[0::2] + rows[1::2]  # Laplace rows, then Stokes rows
scal_tab = grid(
    [9, 9, 7, 11, 7, 11, 9, 13],
    [['machine', 'kernel', 'cores', '1 thread', None, 'full node', None, 'speedup'],
     [None, None, None, 'setup (s)', 'pts/s', 'setup (s)', 'pts/s', None]], [[r] for r in rows], {3: 2, 5: 2},
    """*setup* is the total wall time of the self- and near-interaction setup, single layer
   only, at order 12, 8 patches per face (384 elements, :math:`N=55{,}296` nodes), twist
   :math:`\\pi/6`, tol :math:`10^{-9}`. The double-layer setup costs the same to within a few percent
   at every thread count on all three machines, so building both operators takes about twice the
   times listed.""")

if len(sys.argv) > 1:
    p = pathlib.Path(sys.argv[1])
    rst = p.read_text()
    for start, end, new in [('.. table:: Order 12, 12 patches per face', 'The larger errors at', conv_tab),
                            ('.. table:: *setup* is the total wall time', '- **Per core at one thread**', scal_tab)]:
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
print('\n== quoted quantities (Duffy)')
print('laplace pi/6 error: 1e-3 %s -> 1e-12 %s' % (R[('laplace', '0.5236', '1e-03')]['greens_sol'], R[('laplace', '0.5236', '1e-12')]['greens_sol']))
print('laplace 1e-12 error: pi %s vs pi/6 %s, ratio %.0f' % (R[('laplace', '3.1416', '1e-12')]['greens_sol'], R[('laplace', '0.5236', '1e-12')]['greens_sol'],
      float(R[('laplace', '3.1416', '1e-12')]['greens_sol']) / float(R[('laplace', '0.5236', '1e-12')]['greens_sol'])))
print('laplace pi: error-density %s, error at 1e-12 %s' % (R[('laplace', '3.1416', '1e-12')]['greens_den'], R[('laplace', '3.1416', '1e-12')]['greens_sol']))
print('tolerance cost, laplace pi/6 Rome pps/c: 1e-3 %.0f, 1e-12 %.0f, ratio %.2f' % (f('Rome', 'laplace', '0.5236', '1e-03'), f('Rome', 'laplace', '0.5236', '1e-12'),
      f('Rome', 'laplace', '0.5236', '1e-03') / f('Rome', 'laplace', '0.5236', '1e-12')))
for name in ('Rome', 'Genoa', 'Icelake'):
    print('twist cost at 1e-9, laplace %s: pi/6 %.0f -> pi %.0f, ratio %.2f' % (name, f(name, 'laplace', '0.5236', '1e-09'), f(name, 'laplace', '3.1416', '1e-09'),
          f(name, 'laplace', '0.5236', '1e-09') / f(name, 'laplace', '3.1416', '1e-09')))
ratios = [f(name, k, tw, tol) / f('Rome', k, tw, tol) for name in ('Genoa', 'Icelake') for k in ('laplace', 'stokes')
          for tw in ('0.5236', '1.5708', '3.1416') for tol in ('1e-03', '1e-06', '1e-09', '1e-12')]
print('machine/Rome pps/c ratio over %d comparisons: median %.2f, min %.2f, max %.2f' % (len(ratios), statistics.median(ratios), min(ratios), max(ratios)))
for name in ('Genoa', 'Icelake'):
    rr = [f(name, k, tw, tol) / f('Rome', k, tw, tol) for k in ('laplace', 'stokes') for tw in ('0.5236', '1.5708', '3.1416') for tol in ('1e-03', '1e-06', '1e-09', '1e-12')]
    print('  %s/Rome: median %.2f, min %.2f, max %.2f' % (name, statistics.median(rr), min(rr), max(rr)))
print('\n== scaling (Duffy, tol 1e-9, ppf 8)')
for name, _, cores in MACHINES:
    for kernel in ('laplace', 'stokes'):
        s = scal[name][kernel]
        t = {int(r['thr']): float(r['setup_sl']) for r in s}
        pc = lambda thr: N_SCALING / t[thr] / thr
        e = lambda thr: 100 * t[1] / t[thr] / thr
        print('%-8s %-8s pts/s 1thr %.0f  full %.0f  per-core 1thr %.0f -> full %.0f (%+.0f%%)  eff full %.0f%%  eff@64 %.0f%%  eff@32 %.0f%%' % (
            name, kernel, N_SCALING / t[1], N_SCALING / t[cores], pc(1), pc(cores), 100 * (pc(cores) / pc(1) - 1), e(cores), e(64), e(32)))
