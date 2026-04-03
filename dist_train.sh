#!/bin/bash
config=config/train.py
gpus=$(nvidia-smi --list-gpus | wc -l)

# 4 GPUs required to reproduce the paper results; 
# remove this check if you know what you are doing
if [ "$gpus" -ne 4 ]; then
  echo "Error: expected 4 GPUs, found $gpus." >&2
  exit 1
fi

master_port=$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))          # 0 => pick an available port
print(s.getsockname()[1])
s.close()
PY
)

torchrun \
  --nproc_per_node="$gpus" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port="$master_port" \
  train.py \
  --config "$config" \
  "$@"