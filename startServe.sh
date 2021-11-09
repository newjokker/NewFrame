#!/bin/bash

MODEL_TYPES=${MODEL_TYPES:-M1,M2,M3,M4,M5,M6,M7,M8,M9}
echo $MODEL_TYPES

# shellcheck disable=SC2164
cd /v0.0.1

startServe(){
    python /v0.0.1/scripts/allflow_wuhan.py
}

source /etc/profile

startServe

