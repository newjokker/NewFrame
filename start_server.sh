#!/bin/bash

echo "start serve"

model_list=${model_list:-fzc,fzcRust,kkxTC,kkxClearance,kkxQUiting,jyzZB,nc}
gpu_id_list=${gpu_id_list:-0}
mul_process_num=${mul_process_num:-2}

python3 /v0.0.1/scripts/allflow_wuhan.py --mul_process_num "$mul_process_num" --gpu_id_list "$gpu_id_list" --model_list "$model_list"

echo "stop serve"

