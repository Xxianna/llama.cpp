#!/bin/bash
# pc2 (no GPU) campaign, runs ON pc2. CPU-only builds of b11399 and the release commit + CPU decode benchmarks (Q2_0 kernels, THP, threads).
D=$HOME/campaign; mkdir -p $D; L=$D/log.txt; R=$D/results_pc2.tsv; DEADLINE=$1
log() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a $L; }
left() { echo $(( DEADLINE - $(date +%s) )); }
M=$HOME/qwen36.gguf; cd $HOME/pp || exit 1
git fetch -q neurall release rc-next 2>&1 | tail -1
build() { # dir rev
  [ -d $1 ] || git worktree add -q --detach $1 $2 2>&1 | tail -1
  ( cd $1 && git checkout -q $2 && cmake -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > $D/cfg_$(basename $1).log 2>&1 && nice cmake --build build-cpu --target llama-bench test-quantize-fns -j12 > $D/build_$(basename $1).log 2>&1 ) && log "built $1 $2" || log "BUILD FAILED $1"
}
build $HOME/w-b11399 5c739a3; build $HOME/w-rc 59b14a88c
bench() { # tag bindir threads env
  local out; out=$(cd $2 && env $4 build-cpu/bin/llama-bench -m $M -t $3 -p 512 -n 128 -r 3 -o csv 2>/dev/null | tail -n +2)
  echo "$out" | awk -F, -v t=$1 -v th=$3 '{gsub(/"/,"",$0); n=split($0,a,","); print "RESULT\tpc2\tcpu\t" t "_t" th "\t" (a[n-3] ~ /pp/ ? "pp512" : "tg128") "\t" a[n-1] "\t"}' >> $R
}
for r in 1 2 3; do
  [ $(left) -gt 900 ] || break
  bench old $HOME/w-b11399 6 X=1; bench new $HOME/w-rc 6 X=1; bench new_nothp $HOME/w-rc 6 GGML_HUGEPAGES=0
done
for t in 4 5 12; do [ $(left) -gt 600 ] && { bench new $HOME/w-rc $t X=1; bench old $HOME/w-b11399 $t X=1; }; done
( cd $HOME/w-rc && build-cpu/bin/test-quantize-fns > $D/quantize_fns.txt 2>&1; tail -3 $D/quantize_fns.txt | tee -a $L )
log "pc2 campaign finished"; echo PC2_CAMPAIGN_DONE >> $L
