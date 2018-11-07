 export CUDA_VISIBLE_DEVICES=$1
 python -u -m main \
 --mode=train \
 --epochs=30 \
 --early_stop=5 \
 --model_name=$2'_'base \
 --model=$2 \
 --task=kvret \
 --slm_weight=0 \
 --new_model
