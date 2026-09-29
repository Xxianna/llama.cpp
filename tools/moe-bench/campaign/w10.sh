#!/bin/bash
# W10 campaign (driven from PC1 over ssh). Env: RES DEADLINE
export MACHINE=w10; HERE=$(cd "$(dirname "$0")" && pwd); LIBDIR=$HERE/../overnight; export LIBDIR
source "$LIBDIR/lib.sh"
W=o@192.168.1.2; OLD='N:\rel\b11399'; NEW='N:\rel\b11493'; export OUT=$RES/w10_reg
result_w10() { printf 'RESULT\tw10\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "${5:-}" >> "$RES/results_w10.tsv"; }
idle() { local q=0 t0; t0=$(date +%s); while [ $(( $(date +%s)-t0 )) -lt 5400 ]; do n=$(ssh -o ConnectTimeout=5 $W 'tasklist /NH' 2>/dev/null | grep -ciE "llama-server|llama-perplexity|test-backend|cl.exe|nvcc|ninja"); if [ "${n:-1}" = 0 ]; then q=$((q+1)); [ $q -ge 3 ] && return 0; else q=0; fi; sleep 20; done; }
cmp() { # name budget ARGS_NEW ENV_NEW ARGS_OLD ENV_OLD rounds
  local name=$1; shift; mkdir -p "$OUT"; rm -f "$OUT/raw.txt"
  ARGS_NEW=$1 ENV_NEW=$2 ARGS_OLD=$3 ENV_OLD=$4 bash $HERE/../w10_reg.sh "$NEW" "$OLD" ${5:-2} > "$RES/logs/w10_$name.txt" 2>&1
  python3 - "$OUT/raw.txt" "$name" <<'PY'
import sys,re,json,os
raw,name=sys.argv[1],sys.argv[2]
res=os.environ.get("RES")
for l in open(raw):
    m=re.match(r"(\d+) (\w+) (\w+) (\{.*\})",l)
    if m:
        try: j=json.loads(m.group(4))
        except Exception: continue
        k=m.group(3); v=j["pp"] if k=="long" else j["tg"]
        open(res+"/results_w10.tsv","a").write(f"RESULT\tw10\t{name}\t{m.group(2)}#{m.group(1)}\t{k}\t{v}\t\n")
PY
}
exp_prep() { ssh $W 'powershell -c "if(-not (Test-Path N:\rel\b11493)){Expand-Archive N:\llama.cpp\release-out\llama-release-b11493-59b14a8-bin-win-cuda-13.4-x64.zip N:\rel\b11493}; Test-Path N:\rel\b11493\llama-server.exe"' | tr -d '\r' | grep -q True; }
exp_reg() { cmp reg "" "" "" "" 3; }
exp_predict() { cmp predict "--moe-predict 4 --lrn-prd 32" "" "" "" 3; }
exp_defer() { cmp defer "" "LLAMA_MOE_DEFER=4" "" "" 2; cmp defer_m10 "" "LLAMA_MOE_DEFER=4" "" "" 2; }
exp_ccache() { # clean tree state, ccache launchers, full rebuild timed, then one-file rebuild timed
  ssh $W 'cd /d N:\llama.cpp & git checkout -q -- ggml/src/ggml-cuda/cpy.cu ggml/src/ggml-cuda/concat.cu & echo ok' > /dev/null 2>&1
  printf '@echo off\r\nset TEMP=N:\\tmp\r\nset TMP=N:\\tmp\r\nset CCACHE_DIR=N:\\ccache\r\nset CCACHE_BASEDIR=N:\\llama.cpp\r\ncall N:\\vs\\VC\\Auxiliary\\Build\\vcvars64.bat >nul 2>&1\r\ncd /d N:\\llama.cpp\r\ncmake -B build-cuda-dl -DCMAKE_C_COMPILER_LAUNCHER=N:/ccache/ccache.exe -DCMAKE_CXX_COMPILER_LAUNCHER=N:/ccache/ccache.exe -DCMAKE_CUDA_COMPILER_LAUNCHER=N:/ccache/ccache.exe > N:\\tmp\\cc_cfg.log 2>&1\r\ncmake --build build-cuda-dl --target clean > nul 2>&1\r\necho START %%TIME%% > N:\\tmp\\cc_time.txt\r\ncmake --build build-cuda-dl -j 16 > N:\\tmp\\cc_build.log 2>&1\r\necho FULL_DONE %%TIME%% >> N:\\tmp\\cc_time.txt\r\ncopy /y ggml\\src\\ggml-cuda\\cpy.cu ggml\\src\\ggml-cuda\\cpy.cu.bak > nul\r\ncopy /b ggml\\src\\ggml-cuda\\cpy.cu +,, > nul\r\ncmake --build build-cuda-dl -j 16 > N:\\tmp\\cc_build2.log 2>&1\r\necho ONE_DONE %%TIME%% >> N:\\tmp\\cc_time.txt\r\nccache -s > N:\\tmp\\cc_stats.txt 2>&1\r\n' > /tmp/cc_w10.bat
  scp -q /tmp/cc_w10.bat $W:N:/tmp/cc_w10.bat && timeout 7000 ssh $W 'N:\tmp\cc_w10.bat' > /dev/null 2>&1
  ssh $W 'type N:\tmp\cc_time.txt & findstr /C:hit /C:miss N:\tmp\cc_stats.txt' 2>&1 | tr -d '\r' | head -8 > "$RES/logs/w10_ccache.txt"; log "w10 ccache: $(tr '\n' ' ' < "$RES/logs/w10_ccache.txt")"; }
log "w10 campaign start"; mkdir -p "$RES/logs"
idle; exp prep 900 exp_prep || log "w10 prep failed"
for e in "reg 4500 exp_reg" "predict 4500 exp_predict" "defer 4500 exp_defer" "ccache 4800 exp_ccache"; do set -- $e; [ -e "$RES/STOP" ] && break; idle; exp "$1" "$2" "$3"; done
log "w10 campaign finished"; echo W10_CAMPAIGN_DONE >> "$RES/log.txt"
