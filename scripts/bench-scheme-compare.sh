#!/bin/bash
# Scheme-comparison sweep + OpenMP strong scaling for bench-scheme-compare on one Icelake node.
#
# Compares the four singular-quadrature schemes {RP, Adaptive, Hybrid, Duffy} against each other on
# a single machine. The whole node is allocated to one task and split internally, so both studies
# run from one submission:
#
#   Phase 1 (convergence): the (kernel x scheme) jobs are dispatched across NSLOTS concurrent slots,
#     each pinned to its own disjoint set of PHYSICAL cores (taskset mask + OMP_PLACES=cores). A
#     dynamic flock work-queue hands the next job to whichever slot frees up first, so one slow job
#     (e.g. RectPolar at large Nbeta) never blocks the others.
#   Phase 2 (OpenMP scaling): each (kernel x scheme) runs in turn on ALL physical cores; the binary
#     descends the thread width from the full node down to 1, emitting one row per width.
#
# All raw output goes to a scratch directory and is deleted at the end -- the only artifact left
# behind is the results table scheme-compare.tex in the repository root.
#
#   sbatch scripts/bench-scheme-compare.sh
#
# Watch with `squeue -u $USER`; check the achieved efficiency afterwards with `seff <jobid>`.
#
# NOTE ON -march=native. The Makefile compiles with the default -march=native, so the binary is
# tuned to the CPU that built it. Build and run on the same architecture (this job pins to Icelake
# via --constraint below); a binary built here will SIGILL if run on a different-ISA machine.

#SBATCH --job-name=scheme-compare
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=03:30:00
#SBATCH --output=/tmp/scheme-compare-%j.out
#SBATCH --error=/tmp/scheme-compare-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT"

source ./sctl_source

NSLOTS=2                       # concurrent convergence slots (each gets NPHYS/NSLOTS cores)

echo "host       : $(hostname)"
echo "cpu        : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
echo "sockets    : $(lscpu | awk -F: '/^Socket\(s\)/{print $2}' | tr -d ' ')"
echo "avx512f    : $(grep -c -m1 avx512f /proc/cpuinfo || true)"
echo

make -j16 bin/bench-scheme-compare
BIN="./bin/bench-scheme-compare"

# --- physical-core list that respects the Slurm cgroup: expand the logical CPUs this step is
#     actually allowed to use (/proc/self/status), then keep one logical CPU per physical core. ---
expand_list() { local x; for x in ${1//,/ }; do case $x in *-*) seq "${x%-*}" "${x#*-}";; *) echo "$x";; esac; done; }
ALLOWED_RAW="$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)"
declare -A CORE_OF
while IFS=, read -r cpu core _; do CORE_OF[$cpu]=$core; done \
  < <(lscpu -p=CPU,CORE 2>/dev/null | grep -v '^#')
PCPUS=(); declare -A SEEN_CORE
for cpu in $(expand_list "$ALLOWED_RAW" | sort -n); do
  core="${CORE_OF[$cpu]:-$cpu}"
  if [ -z "${SEEN_CORE[$core]:-}" ]; then SEEN_CORE[$core]=1; PCPUS+=("$cpu"); fi
done
NPHYS=${#PCPUS[@]}
ALLCPUS="$(IFS=,; echo "${PCPUS[*]}")"          # one logical CPU per physical core, whole node
WSLOT=$(( NPHYS / NSLOTS ))                      # physical cores per convergence slot
echo "phys cores : $NPHYS available (allowed=$ALLOWED_RAW)"
echo "dispatch   : phase1 $NSLOTS slots x $WSLOT cores ; phase2 full node ($NPHYS cores)"
echo

# Bind OpenMP threads to physical cores; the width per run comes from the binary's last argument.
export OMP_PLACES=cores
export OMP_PROC_BIND=close

KERNELS="laplace stokes"
SCHEMES="RP Adaptive Hybrid Duffy"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/scheme-compare.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# ================================================================= phase 1: convergence
# Contiguous WSLOT-core taskset list for a given slot (empty if we run out of cores).
slot_cpulist() {
  local i idx start=$(( $1 * WSLOT )) list=""
  for ((i = 0; i < WSLOT; i++)); do
    idx=$(( start + i )); [ "$idx" -lt "$NPHYS" ] || break
    list+="${list:+,}${PCPUS[$idx]}"
  done
  echo "$list"
}

CJOBS=()
for kernel in $KERNELS; do for scheme in $SCHEMES; do CJOBS+=("$kernel $scheme"); done; done
NCJOBS=${#CJOBS[@]}

CTR="$WORK/.next"; echo 0 > "$CTR"
LOCK="$WORK/.lock"; : > "$LOCK"
next_job() {                 # echo the next unclaimed index, or nothing when the queue is drained
  local n
  exec 9>"$LOCK"; flock 9
  n=$(<"$CTR")
  [ "$n" -lt "$NCJOBS" ] && echo $((n + 1)) > "$CTR"
  flock -u 9
  [ "$n" -lt "$NCJOBS" ] && echo "$n"
}
run_slot() {
  local slot=$1 cpulist k kernel scheme out t0
  cpulist="$(slot_cpulist "$slot")"
  while k="$(next_job)"; [ -n "$k" ]; do
    read -r kernel scheme <<< "${CJOBS[$k]}"
    out="$WORK/conv.${kernel}.${scheme}"
    t0=$SECONDS
    echo "[slot $slot cpus=${cpulist:-<all>}] START conv $kernel $scheme"
    taskset -c "${cpulist:-$ALLCPUS}" "$BIN" conv "$kernel" "$scheme" "$WSLOT" \
      > "$out.txt" 2> "$out.err" || echo "  !! conv $kernel $scheme exited nonzero (see $out.err)"
    echo "[slot $slot] DONE  conv $kernel $scheme  ($(( SECONDS - t0 )) s)"
  done
}
echo "### phase 1: convergence sweep ($NCJOBS jobs across $NSLOTS slots) ###"
for ((s = 0; s < NSLOTS; s++)); do run_slot "$s" & done
wait

# ================================================================= phase 2: OpenMP scaling
echo "### phase 2: OpenMP strong scaling (full node, one kernel/scheme at a time) ###"
for kernel in $KERNELS; do
  for scheme in $SCHEMES; do
    out="$WORK/omp.${kernel}.${scheme}"
    t0=$SECONDS
    echo "[full node cpus=$ALLCPUS] START omp $kernel $scheme nt_max=$NPHYS"
    taskset -c "$ALLCPUS" "$BIN" omp "$kernel" "$scheme" "$NPHYS" \
      > "$out.txt" 2> "$out.err" || echo "  !! omp $kernel $scheme exited nonzero (see $out.err)"
    echo "DONE  omp $kernel $scheme  ($(( SECONDS - t0 )) s)"
  done
done

# ================================================================= merge + parse -> single .tex
CONV_RAW="$WORK/conv.txt"; OMP_RAW="$WORK/omp.txt"
cat "$WORK"/conv.*.txt > "$CONV_RAW" 2>/dev/null || : > "$CONV_RAW"
cat "$WORK"/omp.*.txt  > "$OMP_RAW"  2>/dev/null || : > "$OMP_RAW"

bash scripts/parse-scheme-compare.sh "$CONV_RAW" "$OMP_RAW" "scheme-compare.tex"
echo
echo "# results : $ROOT/scheme-compare.tex  (all raw output was scratch and has been discarded)"
