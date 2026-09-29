#!/bin/bash
# PC1 (2x3090) unattended campaign. Env: RES DEADLINE MACHINE=pc1 LIBDIR(=../overnight). Results: $RES/results.tsv, log $RES/log.txt
export MACHINE=pc1; HERE=$(cd "$(dirname "$0")" && pwd); LIBDIR=$HERE/../overnight; export LIBDIR
source "$LIBDIR/lib.sh"
WT=/p/bw/llama.cpp/.claude/worktrees; RC=$WT/rc; RELS=/p/bw/rels
GLM=/m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf; QN=/m/q/4/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
MIMO=/m/m/3/MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-00003.gguf; [ -f "$MIMO" ] || MIMO=/m/m/3/MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-00008.gguf
REL=$RELS/b11399-5c739a3; NEW=$RELS/rel-b11493; TXT=$WT/partial-pin/tools/moe-bench/longsrc.cpp
export PORT=8250 LLAMA_MOE_CACHE_PREDICT_FILE=0
busy() { pgrep -x llama-server > /dev/null || pgrep -x llama-perplexity > /dev/null; }
wait_idle() { local q=0 t0; t0=$(date +%s); while [ $(( $(date +%s)-t0 )) -lt 5400 ]; do if busy; then q=0; else q=$((q+1)); [ $q -ge 4 ] && return 0; fi; sleep 15; done; log "WARN box busy 90 min"; }

prepare() { # wait for the release build, second pass (reverted files), install, package tarball
  local t0; t0=$(date +%s)
  while ! grep -q LINUX_BUILD_EXIT "$RES/rel_build2.log" 2>/dev/null; do [ $(( $(date +%s)-t0 )) -gt 7200 ] && { log "WARN release build not done in 2h"; break; }; sleep 30; done
  cd "$RC" && nice cmake --build build-release -j16 > "$RES/logs/rel_pass2.txt" 2>&1 || log "WARN release build pass 2 failed"
  if [ -x "$RC/build-release/bin/llama-server" ]; then
    rm -rf "${NEW:?}"; mkdir -p "$NEW"; cp -a "$RC/build-release/bin/." "$NEW/"
    mkdir -p $RELS/release-out; local n=llama-release-b11493-59b14a8-bin-ubuntu-cuda-13.2-x64
    rm -rf /tmp/$n; mkdir /tmp/$n; cp -a "$NEW/." /tmp/$n/; cp "$RC/LICENSE" /tmp/$n/ 2>/dev/null
    tar -C /tmp -czf $RELS/release-out/$n.tar.gz $n && log "packaged $RELS/release-out/$n.tar.gz"; rm -rf /tmp/$n
    log "NEW = release build: $("$NEW/llama-server" --version 2>&1 | grep -m1 version)"
  else NEW=$RELS/rc1; log "WARN using dev binary rc1 as NEW"; fi
}
ppl_run() { # tag bin envs...
  local tag=$1 bin=$2; shift 2; wait_ram 110 300
  local out="$RES/logs/ppl_$tag.txt"
  env "$@" LLAMA_MOE_CACHE_STATS=1 timeout 1500 "$bin/llama-perplexity" -m $GLM -f $TXT -c 1024 --chunks 2 -b 1 -ub 1 > "$out" 2>&1
  result ppl "$tag" ppl "$(grep -o 'PPL = [0-9.]*' "$out" | tail -1 | grep -o '[0-9.]*$')" "spp=$(grep -o '[0-9.]* seconds per pass' "$out" | head -1)"
}
kl_run() { # tag bin base|kl envs...
  local tag=$1 bin=$2 mode=$3; shift 3; wait_ram 110 300; local out="$RES/logs/kl_${tag}_$mode.txt" a="--kl-divergence-base $RES/kl_base.bin"
  [ $mode = kl ] && a="$a --kl-divergence"
  env "$@" timeout 1500 "$bin/llama-perplexity" -m $GLM -f $TXT -c 1024 --chunks 2 -b 1 -ub 1 $a > "$out" 2>&1
  [ $mode = kl ] && { result kl "$tag" mean_kld "$(grep -o 'Mean    KLD: *[0-9.]*' "$out" | grep -o '[0-9.]*$')"; result kl "$tag" top1 "$(grep -o 'Same top p: *[0-9.]*' "$out" | grep -o '[0-9.]*$')"; }
}
exp_reg() { local r; for r in 1 2 3; do if [ $((r%2)) = 1 ]; then run_variant reg old_b11399#$r $GLM $REL/llama-server "" "short:1,chat:2,12k:1"; run_variant reg new#$r $GLM $NEW/llama-server "" "short:1,chat:2,12k:1"; else run_variant reg new#$r $GLM $NEW/llama-server "" "short:1,chat:2,12k:1"; run_variant reg old_b11399#$r $GLM $REL/llama-server "" "short:1,chat:2,12k:1"; fi; done; }
exp_ppl() { ppl_run old $REL LLAMA_X=1; ppl_run new $NEW LLAMA_X=1; ppl_run new_v16 $NEW GGML_CUDA_MMVF_V16=1; }
exp_v16() { local r; for r in 1 2 3; do run_variant v16 off#$r $GLM $NEW/llama-server "GGML_CUDA_MMVF_V16=0" "chat:3"; run_variant v16 on#$r $GLM $NEW/llama-server "GGML_CUDA_MMVF_V16=1" "chat:3"; done; }
exp_kl() { kl_run base $NEW base LLAMA_DEFER=0; kl_run noise $NEW kl LLAMA_DEFER=0; kl_run v16 $NEW kl GGML_CUDA_MMVF_V16=1; kl_run defer4_m10 $NEW kl LLAMA_MOE_DEFER=4 LLAMA_MOE_DEFER_MASS=0.1; kl_run defer4_m20 $NEW kl LLAMA_MOE_DEFER=4 LLAMA_MOE_DEFER_MASS=0.2; kl_run defer2 $NEW kl LLAMA_MOE_DEFER=2; }
exp_defer() { local r; for r in 1 2; do run_variant defer d0#$r $GLM $NEW/llama-server "LLAMA_MOE_DEFER=0" "chat:3"; run_variant defer m10#$r $GLM $NEW/llama-server "LLAMA_MOE_DEFER=4 LLAMA_MOE_DEFER_MASS=0.1" "chat:3"; run_variant defer m20#$r $GLM $NEW/llama-server "LLAMA_MOE_DEFER=4 LLAMA_MOE_DEFER_MASS=0.2" "chat:3"; run_variant defer d4#$r $GLM $NEW/llama-server "LLAMA_MOE_DEFER=4" "chat:3"; done; }
exp_policy() { run_variant policy default $GLM $NEW/llama-server "LLAMA_MOE_CACHE_STATS=1" "chat:8"; run_variant policy m0_gw2 $GLM $NEW/llama-server "LLAMA_MOE_CACHE_STATS=1 LLAMA_MOE_CACHE_MARGIN=0 LLAMA_MOE_CACHE_GLOBAL_WEIGHT=2" "chat:8"; run_variant policy m0_gw0 $GLM $NEW/llama-server "LLAMA_MOE_CACHE_STATS=1 LLAMA_MOE_CACHE_MARGIN=0 LLAMA_MOE_CACHE_GLOBAL_WEIGHT=0" "chat:8"; }
exp_ub() { run_variant ub default $GLM $NEW/llama-server "" "chat:3"; run_variant ub ub512 $GLM $NEW/llama-server "" "chat:3" -ub 512; run_variant ub ub512_c8k $GLM $NEW/llama-server "" "chat:3" -ub 512 -c 8192; }
exp_qn() { export WAITGB=100; run_variant qn old $QN $REL/llama-server "" "short:2,12k:1"; run_variant qn new $QN $NEW/llama-server "" "short:2,12k:1"; run_variant qn old2 $QN $REL/llama-server "" "short:2,12k:1"; run_variant qn new2 $QN $NEW/llama-server "" "short:2,12k:1"; unset WAITGB; }
exp_mimo() { export WAITGB=20 SRV_TIMEOUT=2400; run_variant mimo old $MIMO $REL/llama-server "" "short:2,12k:1"; run_variant mimo new $MIMO $NEW/llama-server "" "short:2,12k:1"; unset WAITGB SRV_TIMEOUT; }
exp_repeat() { local r=0; while have_time 1500; do r=$((r+1)); run_variant repeat old#$r $GLM $REL/llama-server "" "short:1,chat:2"; run_variant repeat new#$r $GLM $NEW/llama-server "" "short:1,chat:2"; done; }

log "pc1 campaign start, deadline in $(time_left)s"
exp prepare 7500 prepare; wait_idle
for e in "reg 3300 exp_reg" "ppl 1800 exp_ppl" "v16 2400 exp_v16" "kl 3000 exp_kl" "policy 1500 exp_policy" "defer 3600 exp_defer" "ub 1200 exp_ub" "qn 2400 exp_qn" "mimo 2400 exp_mimo" "repeat 1500 exp_repeat"; do
  set -- $e; [ -e "$RES/STOP" ] && { log "STOP file: end"; break; }; wait_idle; exp "$1" "$2" "$3"; python3 "$HERE/report.py" "$RES" > "$RES/REPORT.md" 2>&1
done
log "pc1 campaign finished"; echo PC1_CAMPAIGN_DONE >> "$RES/log.txt"
