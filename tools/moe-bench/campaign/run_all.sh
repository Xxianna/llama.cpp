#!/bin/bash
# run_all.sh [hours]: starts the PC1, W10 and pc2 campaigns detached. Stop everything early: touch $RES/STOP
HERE=$(cd "$(dirname "$0")" && pwd); H=${1:-6}
export RES=/p/bw/data/campaign-0929 DEADLINE=$(( $(date +%s) + H*3600 )); mkdir -p $RES/logs
cp /tmp/claude-1000/*/*/scratchpad/rel_build2.log $RES/rel_build2.log 2>/dev/null
scp -q $HERE/pc2.sh o@192.168.1.12:/tmp/pc2_campaign.sh && ssh o@192.168.1.12 "setsid nohup bash /tmp/pc2_campaign.sh $DEADLINE > /tmp/pc2_campaign.out 2>&1 < /dev/null &" 
setsid nohup bash $HERE/w10.sh > $RES/w10.out 2>&1 < /dev/null &
setsid nohup bash $HERE/pc1.sh > $RES/pc1.out 2>&1 < /dev/null &
# pull pc2 results/logs back every 10 min and refresh the report
setsid nohup bash -c 'while [ $(date +%s) -lt '$DEADLINE' ]; do scp -q o@192.168.1.12:campaign/results_pc2.tsv '$RES'/ 2>/dev/null; scp -q o@192.168.1.12:campaign/log.txt '$RES'/pc2_log.txt 2>/dev/null; python3 '$HERE'/report.py '$RES' > '$RES'/REPORT.md 2>&1; sleep 600; done' > /dev/null 2>&1 < /dev/null &
echo "campaign started $(date), deadline $(date -d @$DEADLINE)"
