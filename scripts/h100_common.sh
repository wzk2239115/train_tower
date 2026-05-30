# Shared env for 8× H100/H800 torchrun training (sourced by h100_*.sh).

h100_env_setup() {
  export NUM_GPUS="${NUM_GPUS:-8}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

  # Prefer IPv4 in K8s/containers (avoids c10d "Address family not supported").
  export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"

  if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
    for nic in eth0 ens5 enp0s8 bond0; do
      if ip -o link show "$nic" &>/dev/null; then
        export NCCL_SOCKET_IFNAME="$nic"
        export GLOO_SOCKET_IFNAME="$nic"
        break
      fi
    done
  fi

  export CONFIG="${CONFIG:-configs/train/world_pt_h800.yaml}"
  export USE_DEEPSPEED="${USE_DEEPSPEED:-1}"
}

h100_run_torchrun() {
  torchrun --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    -m tower.cli train --config "${CONFIG}" "${TRAIN_ENV_EXTRA[@]}" "$@"
}
