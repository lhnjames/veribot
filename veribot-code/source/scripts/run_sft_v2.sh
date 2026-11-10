#!/bin/bash
cd /path/to/project_root && source activate.sh && python -m verifiable_bot.train_sft --data /path/to/data_root/outputs/veriroute_lite/cold_start_sft_v4_final.jsonl --output outputs/veriroute_lite/sft_0.6b_20260722_0252 --model Qwen/Qwen3-0.6B >> outputs/veriroute_lite/sft_0.6b_train_v2.log 2>&1
