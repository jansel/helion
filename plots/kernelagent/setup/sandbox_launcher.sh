#!/bin/bash
set -euo pipefail

RUN_ROOT=${KERNELAGENT_ALLOWED_RUN_ROOT:?KERNELAGENT_ALLOWED_RUN_ROOT is required}
SOURCE_ROOT=${KERNELAGENT_SOURCE_ROOT:-/tmp/kernelagent-study/KernelAgent}
SHIMS_ROOT=${KERNELAGENT_SHIMS_ROOT:-/tmp/kernelagent-study/shims}
PHYSICAL_GPU=${KERNELAGENT_PHYSICAL_GPU:?KERNELAGENT_PHYSICAL_GPU is required}
ENV_ROOT=${KERNELAGENT_ENV_ROOT:-${VIRTUAL_ENV:-}}
: "${ENV_ROOT:?KERNELAGENT_ENV_ROOT or VIRTUAL_ENV is required}"
ENV_ROOT=$(readlink -f "$ENV_ROOT")
HOST_CWD=$PWD
ROOT=$(mktemp -d /tmp/kernelagent-jail.XXXXXX)
trap 'rm -rf "$ROOT"' EXIT

GPU_MINOR=$(nvidia-smi -i "$PHYSICAL_GPU" -q | awk -F: '/Minor Number/{gsub(/ /,"",$2); minor=$2} END{print minor}')
CUDA_TARGET=$(readlink -f /usr/local/cuda)

exec 9>"/tmp/kernelagent-gpu-${PHYSICAL_GPU}.lock"
flock 9

unshare --user --map-root-user --mount --net --pid --fork /bin/bash -s -- \
    "$ROOT" "$RUN_ROOT" "$SOURCE_ROOT" "$SHIMS_ROOT" "$ENV_ROOT" \
    "$HOST_CWD" "$GPU_MINOR" "$CUDA_TARGET" "$@" <<'INNER'
set -euo pipefail
ROOT=$1; shift
RUN_ROOT=$1; shift
SOURCE_ROOT=$1; shift
SHIMS_ROOT=$1; shift
ENV_ROOT=$1; shift
HOST_CWD=$1; shift
GPU_MINOR=$1; shift
CUDA_TARGET=$1; shift

mount --make-rprivate /
mount -t tmpfs -o mode=755,size=16G tmpfs "$ROOT"
mkdir -p "$ROOT"/{dev,etc/alternatives,env,opt/nvidia,proc,run,sys,tmp,usr}

mount --rbind /usr "$ROOT/usr"
mount -o remount,bind,ro "$ROOT/usr"
mount --bind "$ENV_ROOT" "$ROOT/env"
mount -o remount,bind,ro "$ROOT/env"
mount --rbind /opt/nvidia "$ROOT/opt/nvidia"
mount -o remount,bind,ro "$ROOT/opt/nvidia"
mount --rbind /sys "$ROOT/sys"
mount -o remount,bind,ro "$ROOT/sys"

bind_same() {
    local source=$1
    local mode=$2
    local target="$ROOT$source"
    mkdir -p "$target"
    mount --bind "$source" "$target"
    if [[ $mode == ro ]]; then
        mount -o remount,bind,ro "$target"
    fi
}
bind_same "$RUN_ROOT" rw
bind_same "$SOURCE_ROOT" ro
bind_same "$SHIMS_ROOT" ro

for config in ld.so.cache nsswitch.conf passwd group localtime; do
    if [[ -f "/etc/$config" ]]; then
        touch "$ROOT/etc/$config"
        mount --bind "/etc/$config" "$ROOT/etc/$config"
        mount -o remount,bind,ro "$ROOT/etc/$config"
    fi
done
ln -s /usr/bin/ld.bfd "$ROOT/etc/alternatives/ld"
ln -s "$CUDA_TARGET" "$ROOT/etc/alternatives/cuda"

mount -t proc -o nosuid,nodev,noexec proc "$ROOT/proc"
mount -t tmpfs -o mode=755,nosuid tmpfs "$ROOT/dev"
mkdir -p "$ROOT/dev/nvidia-caps" "$ROOT/dev/shm"
mount -t tmpfs -o mode=1777,nosuid,nodev tmpfs "$ROOT/dev/shm"
for device in null zero random urandom nvidiactl nvidia-uvm nvidia-uvm-tools nvidia-modeset "nvidia$GPU_MINOR"; do
    touch "$ROOT/dev/$device"
    mount --bind "/dev/$device" "$ROOT/dev/$device"
done
for cap in /dev/nvidia-caps/*; do
    target="$ROOT/dev/nvidia-caps/$(basename "$cap")"
    touch "$target"
    mount --bind "$cap" "$target"
done
ln -s /proc/self/fd "$ROOT/dev/fd"
ln -s /proc/self/fd/0 "$ROOT/dev/stdin"
ln -s /proc/self/fd/1 "$ROOT/dev/stdout"
ln -s /proc/self/fd/2 "$ROOT/dev/stderr"
ln -s usr/bin "$ROOT/bin"
ln -s usr/lib "$ROOT/lib"
ln -s usr/lib64 "$ROOT/lib64"
ln -s usr/sbin "$ROOT/sbin"

mkdir -p "$RUN_ROOT/.sandbox-home" "$RUN_ROOT/.triton-cache" "$RUN_ROOT/.torch-extensions"

if [[ ${1:-} == /usr/local/cuda/bin/ncu || ${1:-} == ncu ]]; then
    COMMAND=("$@")
else
    COMMAND=(/env/bin/python -S "$@")
fi

chroot "$ROOT" /usr/bin/env -i \
    HOME="$RUN_ROOT/.sandbox-home" \
    PATH=/env/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin \
    PYTHONPATH="$SHIMS_ROOT:$SOURCE_ROOT:/env/lib/python3.12/site-packages" \
    CUDA_VISIBLE_DEVICES=0 \
    CUDA_CACHE_PATH="$RUN_ROOT/.triton-cache/cuda" \
    TRITON_CACHE_DIR="$RUN_ROOT/.triton-cache/triton" \
    TORCH_EXTENSIONS_DIR="$RUN_ROOT/.torch-extensions" \
    /bin/bash -c 'cd "$1"; shift; exec "$@"' bash "$HOST_CWD" "${COMMAND[@]}"
INNER
