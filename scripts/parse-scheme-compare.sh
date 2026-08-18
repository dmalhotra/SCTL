#!/bin/bash
# Parse bench-scheme-compare output into text tables + one compilable LaTeX file (pure bash + gawk).
#
# bin/bench-scheme-compare emits one machine-readable line per configuration:
#   @@ROW kernel=<k> scheme=<s> thr=<n> twist=<t> tol=<tol> error=<e> pps=<p> setup=<s>
# In a convergence run twist/tol vary at a fixed thread width; in an OpenMP run the thread width is
# swept at a fixed twist/tol. This parser consumes both raw logs and compares the four schemes
# {RP, Adaptive, Hybrid, Duffy} side by side.
#
# Two tables are produced:
#   1. convergence      -- Green's-identity error and single-layer setup throughput per scheme,
#                          over a twist x tolerance grid.
#   2. OpenMP scaling   -- per scheme, one row per thread width (setup time, throughput, speedup,
#                          efficiency).
#
# Usage:
#   parse-scheme-compare.sh <conv.txt> <omp.txt> [out.tex]
# Prints both tables to stdout and, if an output path is given, writes a single standalone
# <out.tex> (pdflatex-ready) holding both tables. A missing / '-' input skips that table.
set -euo pipefail

CONV="${1:-}"
OMP="${2:-}"
OUT="${3:-}"

SCHEMES="RP Adaptive Hybrid Duffy"

# ---------------------------------------------------------------- shared gawk library (functions)
read -r -d '' AWKLIB <<'AWK' || true
function sci_parts(x,   e,m) {           # -> "mant|exp" for x>0, else ""
  if (x <= 0 || x != x) return ""
  e = int(log(x)/log(10)); m = x/(10^e)
  while (m >= 10) { m /= 10; e++ }
  while (m <  1) { m *= 10; e-- }
  return sprintf("%.2f|%d", m, e)
}
function sci_txt(x,   p,a) { p=sci_parts(x); if(p=="")return "--"; split(p,a,"|"); return sprintf("%.2fe%+03d",a[1],a[2]) }
function sci_tex(x,   p,a) { p=sci_parts(x); if(p=="")return "--"; split(p,a,"|"); return sprintf("$%.2f\\times10^{%d}$",a[1],a[2]) }
function twlabel(tw, latex,   j) {
  for (j=1; j<=ntw; j++) if (tw-tv[j] < 1e-3 && tv[j]-tw < 1e-3) return latex ? tx[j] : tl[j]
  return sprintf("%.3f", tw)
}
function twidx(tw,   j) { for (j=1;j<=ntw;j++) if (tw-tv[j] < 1e-3 && tv[j]-tw < 1e-3) return j; return 0 }
function parse_row(   i,kv) {            # fills global d[] from an @@ROW line ($0)
  delete d
  for (i=1;i<=NF;i++) if (split($i,kv,"=")==2) d[kv[1]]=kv[2]
}
BEGIN {
  ns = split(SCHEMELIST, S, " ")
  PI = atan2(0,-1)
  tv[1]=PI/6; tl[1]="pi/6"; tx[1]="\\pi/6"
  tv[2]=PI/2; tl[2]="pi/2"; tx[2]="\\pi/2"
  tv[3]=PI;   tl[3]="pi";   tx[3]="\\pi"
  ntw = 3
}
AWK

# ---------------------------------------------------------------- convergence gawk program
# MODE=text  -> stdout table ; MODE=texbody -> a \begin{table}..\end{table} fragment (no preamble)
read -r -d '' AWK_CONV <<'AWK' || true
/^@@ROW/ {
  parse_row()
  k=d["kernel"]; s=d["scheme"]; tw=d["twist"]+0; tol=d["tol"]+0
  ti=twidx(tw); if (ti==0) next
  if (!(k in kseen)) { kseen[k]=++nk; korder[nk]=k }
  tolkey=sprintf("%.3e",tol)
  if (!(tolkey in tolseen)) { tolseen[tolkey]=tol; TVraw[++nt]=tol }
  twpresent[k SUBSEP ti]=1
  key=k SUBSEP ti SUBSEP tolkey SUBSEP s
  E[key]=d["error"]+0; P[key]=d["pps"]+0; H[key]=1
}
END {
  # tols descending
  n = asort(TVraw, TVs); for (i=1;i<=n;i++) TOL[i]=TVs[n-i+1]

  if (MODE=="text") {
    printf "%-14s %7s | %-5s", "kernel,twist", "tol", "error"
    for (j=1;j<=ns;j++) printf " %9s", S[j]
    printf " | %-10s", "pts/s/core"
    for (j=1;j<=ns;j++) printf " %8s", S[j]
    printf "\n"
    for (c=0;c<118;c++) printf "-"; printf "\n"
    for (ki=1;ki<=nk;ki++) { k=korder[ki]
      for (ti=1;ti<=ntw;ti++) { if (!((k SUBSEP ti) in twpresent)) continue
        first=1
        for (i=1;i<=n;i++) { tol=TOL[i]; tk=sprintf("%.3e",tol)
          lbl = first ? (k ", " twlabel(tv[ti],0)) : ""
          first=0
          printf "%-14s %7.0e | %5s", lbl, tol, ""
          for (j=1;j<=ns;j++) { key=k SUBSEP ti SUBSEP tk SUBSEP S[j]
            printf " %9s", (key in H) ? sci_txt(E[key]) : "--" }
          printf " | %10s", ""
          for (j=1;j<=ns;j++) { key=k SUBSEP ti SUBSEP tk SUBSEP S[j]
            printf " %8s", (key in H) ? sprintf("%.0f",P[key]) : "--" }
          printf "\n"
        }
        printf "\n"
      }
    }
  } else {   # texbody
    print "\\begin{table}[t]\\centering\\small"
    print "\\setlength{\\tabcolsep}{4pt}"
    spec="l l"; for (j=1;j<=ns;j++) spec=spec " r"; for (j=1;j<=ns;j++) spec=spec " r"
    print "\\begin{tabular}{" spec "}"
    print "\\toprule"
    printf "& & \\multicolumn{%d}{c}{error} & \\multicolumn{%d}{c}{pts/s/core} \\\\\n", ns, ns
    printf "\\cmidrule(lr){3-%d} \\cmidrule(lr){%d-%d}\n", 2+ns, 3+ns, 2+2*ns
    printf "kernel, twist & tol"
    for (j=1;j<=ns;j++) printf " & %s", S[j]
    for (j=1;j<=ns;j++) printf " & %s", S[j]
    print " \\\\"
    print "\\midrule"
    fb=1
    for (ki=1;ki<=nk;ki++) { k=korder[ki]
      for (ti=1;ti<=ntw;ti++) { if (!((k SUBSEP ti) in twpresent)) continue
        if (!fb) print "\\addlinespace"; fb=0
        first=1
        for (i=1;i<=n;i++) { tol=TOL[i]; tk=sprintf("%.3e",tol)
          lbl = first ? (k ", $" twlabel(tv[ti],1) "$") : ""
          first=0
          printf "%s & %s", lbl, sci_tex(tol)
          for (j=1;j<=ns;j++) { key=k SUBSEP ti SUBSEP tk SUBSEP S[j]
            printf " & %s", (key in H) ? sci_tex(E[key]) : "--" }
          for (j=1;j<=ns;j++) { key=k SUBSEP ti SUBSEP tk SUBSEP S[j]
            printf " & %s", (key in H) ? sprintf("%.0f",P[key]) : "--" }
          print " \\\\"
        }
      }
    }
    print "\\bottomrule"
    print "\\end{tabular}"
    print "\\caption{On-surface Green's identity error and single-layer setup throughput on a twisted cubed sphere (order 12, 12 patches/face), by quadrature scheme.}"
    print "\\end{table}"
  }
}
AWK

# ---------------------------------------------------------------- omp gawk program
# Per-scheme strong scaling: one section (text) / block (tex) per (kernel, scheme), one row per
# thread width. setup(s), pts/s/core (the binary's per-core throughput), pts/s tot (= pps*thr),
# speedup (= setup(1)/setup(thr)), eff (= speedup/thr). error is a per-width sanity check.
read -r -d '' AWK_OMP <<'AWK' || true
/^@@ROW/ {
  parse_row()
  k=d["kernel"]; s=d["scheme"]; thr=d["thr"]+0
  if (!(k in kseen))   { kseen[k]=++nk; korder[nk]=k }
  ks=k SUBSEP s
  ksseen[ks]=1
  key=ks SUBSEP thr
  SET[key]=d["setup"]+0; PPS[key]=d["pps"]+0; ERR[key]=d["error"]+0; HAVE[key]=1
  if (!(thr in thrseen)) { thrseen[thr]=1; THR[++nthr]=thr }
}
END {
  nw = asort(THR, TW)   # thread widths ascending
  if (MODE=="text") {
    for (ki=1;ki<=nk;ki++) { k=korder[ki]
      for (si=1;si<=ns;si++) { s=S[si]; ks=k SUBSEP s
        if (!(ks in ksseen)) continue
        printf "=== %s / %s ===\n", k, s
        printf "%4s  %10s  %11s  %12s  %9s  %7s  %11s\n",
          "thr","setup(s)","pts/s/core","pts/s_tot","speedup","eff","error"
        printf "%4s  %10s  %11s  %12s  %9s  %7s  %11s\n",
          "----","----------","-----------","------------","---------","-------","-----------"
        base=""
        for (wi=1;wi<=nw;wi++) { thr=TW[wi]; key=ks SUBSEP thr
          if (!(key in HAVE)) continue
          st=SET[key]; pc=PPS[key]; tot=pc*thr; er=ERR[key]
          if (base=="" && st>0) base=st
          spd=(base!="" && st>0)?base/st:0; eff=(thr>0)?spd/thr:0
          printf "%4d  %10.4f  %11.1f  %12.1f  %8.2fx  %6.2f%%  %11s\n",
            thr, st, pc, tot, spd, eff*100, sci_txt(er)
        }
        printf "\n"
      }
    }
  } else {   # texbody
    print "\\begin{table}[t]\\centering\\small"
    print "\\setlength{\\tabcolsep}{6pt}"
    print "\\begin{tabular}{l l r r r r r}"
    print "\\toprule"
    print "kernel, scheme & thr & setup (s) & pts/s/core & pts/s tot & speedup & eff \\\\"
    print "\\midrule"
    fb=1
    for (ki=1;ki<=nk;ki++) { k=korder[ki]
      for (si=1;si<=ns;si++) { s=S[si]; ks=k SUBSEP s
        if (!(ks in ksseen)) continue
        if (!fb) print "\\addlinespace"; fb=0
        base=""; first=1
        for (wi=1;wi<=nw;wi++) { thr=TW[wi]; key=ks SUBSEP thr
          if (!(key in HAVE)) continue
          st=SET[key]; pc=PPS[key]; tot=pc*thr
          if (base=="" && st>0) base=st
          spd=(base!="" && st>0)?base/st:0; eff=(thr>0)?spd/thr:0
          lbl = first ? (k ", " s) : ""; first=0
          printf "%s & %d & %.4f & %.0f & %.0f & %.2f$\\times$ & %.1f\\%% \\\\\n",
            lbl, thr, st, pc, tot, spd, eff*100
        }
      }
    }
    print "\\bottomrule"
    print "\\end{tabular}"
    print "\\caption{OpenMP strong scaling of the single-layer setup on a twisted cubed sphere (order 12, 8 patches/face, twist $\\pi/6$, tol $10^{-9}$), one block per quadrature scheme. speedup $=$ setup(1 thread)$/$setup($n$), eff $=$ speedup$/n$.}"
    print "\\end{table}"
  }
}
AWK

have() { [ -n "$1" ] && [ "$1" != "-" ] && [ -f "$1" ]; }

echo "========================================================================"
echo "CONVERGENCE (error + pts/s/core per scheme)"
echo "========================================================================"
if have "$CONV"; then gawk -v SCHEMELIST="$SCHEMES" -v MODE=text "$AWKLIB$AWK_CONV" "$CONV"
else echo "# (no convergence input)"; fi

echo
echo "========================================================================"
echo "OPENMP STRONG SCALING (one section per kernel/scheme)"
echo "========================================================================"
if have "$OMP"; then gawk -v SCHEMELIST="$SCHEMES" -v MODE=text "$AWKLIB$AWK_OMP" "$OMP"
else echo "# (no omp input)"; fi

# --- single combined standalone LaTeX file (both tables) ---
if [ -n "$OUT" ]; then
  {
    echo "\\documentclass[11pt]{article}"
    echo "\\usepackage{booktabs}"
    echo "\\usepackage[margin=1in,landscape]{geometry}"
    echo "\\begin{document}"
    if have "$CONV"; then gawk -v SCHEMELIST="$SCHEMES" -v MODE=texbody "$AWKLIB$AWK_CONV" "$CONV"; fi
    if have "$OMP";  then gawk -v SCHEMELIST="$SCHEMES" -v MODE=texbody "$AWKLIB$AWK_OMP"  "$OMP";  fi
    echo "\\end{document}"
  } > "$OUT"
  echo
  echo "# wrote $OUT"
fi
