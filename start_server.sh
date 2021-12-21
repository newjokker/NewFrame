#!/bin/bash

echo "start serve"

model_list=${model_list:-fzc,fzcRust,kkxTC,kkxClearance,kkxQUiting,jyzZB,nc}
gpu_id_list=${gpu_id_list:-0}
mul_process_num=${mul_process_num:-2}
MODEL_TYPES=${MODEL_TYPES:-00,01,02,03,04,05,06}


python3 /v0.0.1/scripts/allflow_wuhan.py --mul_process_num "$mul_process_num" --gpu_id_list "$gpu_id_list" --model_list "$model_list" --model_types "$MODEL_TYPES"

echo "stop serve"

