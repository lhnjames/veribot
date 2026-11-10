#!/bin/bash
cd /path/to/project_root
source activate.sh
python -m scripts.eval_twibot20_official_shard \
  --checkpoint outputs/veriroute_lite/grpo_0.6b/final --base-model outputs/veriroute_lite/sft_0.6b/merged_rebuilt --test-json /path/to/data_root/data/twibot20_complete/test.json \
  --output outputs/veriroute_lite/eval_twibot20_official/shard_0.jsonl --num-shards 6 --shard-index 0 \
  >> outputs/veriroute_lite/eval_twibot20_official/shard_0.log 2>&1
