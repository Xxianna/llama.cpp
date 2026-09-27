#!/bin/bash
# round-robin runner: run_all.sh <data> "<algo args>" ... ; appends generated-token lines to results.txt
export CUDA_VISIBLE_DEVICES=1
data=$1; shift
for cmd in "$@"; do
  echo "## $data: $cmd" >> results.txt
  python3 $cmd --data $data 2>&1 | grep -E "generated only|Error|error" >> results.txt
done
