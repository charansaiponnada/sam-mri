#!/usr/bin/env bash
# Book-chapter experiments for "Adapting Promptable Foundation Models to Brain MRI: SAM-Style
# Segmentation on BraTS" (MSA Task01 pilot data).
#
#   STAGE=data|zero|medsam|train|prompt|report   (default: all)
#   SEEDS=1,2,3      (training seeds, every trained arm)
#
# Workload: two NVIDIA L40S cards (CUDA_VISIBLE_DEVICES 0/1), launched in parallel for STAGE=all.
# Cheap-first per arm: zero-shot eval -> decoder -> LoRA -> bottleneck -> 2.5D -> 3D -> full FT.
set -euo pipefail
export PYTHONPATH=src
STAGE=${STAGE:-all}
SEEDS=${SEEDS:-1,2,3}
CH=configs/chapter

run_train() { # $1 gpu  $2 arm  $3 seed  [extra --set ...]
  local gpu=$1 arm=$2 seed=$3; shift 3
  local name="ch_${arm}_s${seed}"
  echo "== [gpu $gpu] train ${name}"
  CUDA_VISIBLE_DEVICES=$gpu python -m samri.train --config $CH/${arm}.yaml --set name=${name} seed=${seed} "$@"
  echo "== [gpu $gpu] eval  ${name}"
  CUDA_VISIBLE_DEVICES=$gpu python -m samri.eval --config $CH/${arm}.yaml --set name=${name} seed=$seed --ckpt runs/${name}/best.pt "$@"
}

run_eval_only() { # $1 gpu  $2 arm  $3 seed  $4 ckpt (none = zero-shot / medsam)
  local gpu=$1 arm=$2 seed=$3 ckpt=$4
  echo "== [gpu $gpu] eval  ${arm}_s${seed}"
  if [ "$ckpt" != "none" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python -m samri.eval --config $CH/${arm}.yaml --set name=${arm}_s${seed} seed=$seed --ckpt "$ckpt" "$@"
  else
    CUDA_VISIBLE_DEVICES=$gpu python -m samri.eval --config $CH/${arm}.yaml --set name=${arm}_s${seed} seed=$seed "$@"
  fi
}

stage_data() {
  python -m samri.data.stream --raw data/raw/MSD --out data/msd_meta --layout brats21 \
    --workers 8 --size 256 --cache data/msd_cache
}

stage_zero() { run_eval_only 0 ch_zero_shot 1 none; }
stage_medsam() { run_eval_only 1 ch_medsam 1 checkpoints/medsam_vit_b.pth || echo "medsam ckpt missing -> MedSAM row = not run"; }

# cheap-first arms per seed (each arm ~= 2 train epochs on the 25% patient budget)
stage_train_seed() { # $1 = seed, $2 = gpu
  local seed=$1 gpu=$2
  for arm in ch_decoder ch_lora ch_bottleneck ch_25d ch_3d ch_fullft; do
    run_train $gpu $arm $seed
  done
}

stage_prompt() { run_eval_only 0 ch_lora_prompts 1 runs/ch_lora_s1/best.pt; }
stage_report() {
  python -m samri.report tables --runs 'runs/ch_*'
  python -m samri.report figures --runs 'runs/ch_*'
  python -m samri.failure --config configs/chapter/lora.yaml \
    --ckpt runs/ch_lora_s1/best.pt --run ch_lora_s1 --k 3
}

case ${STAGE} in
  data)   stage_data ;;
  zero)   stage_zero ;;
  medsam) stage_medsam ;;
  train)  stage_train_seed 1 ;;
  prompt) stage_prompt ;;
  report) stage_report ;;
  all)
    stage_data &            # gpu0, cheap: streaming setup + zero-shot eval
    stage_medsam &           # gpu1, cheap: MedSAM eval (or not-run)
    stage_train_seed 1 0 &   # gpu0: decoder..fullft, seed 1
    stage_train_seed 2 1 &   # gpu1: decoder..fullft, seed 2
    stage_train_seed 3 1 &   # gpu1: decoder..fullft, seed 3
    wait
    stage_prompt
    stage_report
    ;;
  *) echo "unknown stage: ${STAGE}" >&2; exit 2 ;;
esac
echo "STAGE ${STAGE} DONE"
