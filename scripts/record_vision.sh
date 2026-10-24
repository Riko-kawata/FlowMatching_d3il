#!/bin/bash
# 画像入力 3 条件だけ (record_all.sh の vision 分の再実行用)。Usage (experiments/ から): bash dl_kouza/scripts/record_vision.sh
PY=/mnt/data1/home/rkawata/miniconda3/envs/d3il_env/bin/python
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
OUT=dl_kouza/videos
mkdir -p $OUT/logs
run() {  # gpu task mod ta
  CUDA_VISIBLE_DEVICES=$1 $PY dl_kouza/scripts/record_flow_videos.py --task $2 --mod $3 --ta $4 \
    --seed 0 --contexts 0 1 2 3 4 --out $OUT > $OUT/logs/$2_$3.log 2>&1
}
run 1 sorting_4 vision 2 &
run 3 sorting_6 vision 2 &
run 5 stacking vision 2 &
wait
