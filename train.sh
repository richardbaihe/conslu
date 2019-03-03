# $1=cuda $2=model $3=weight $4=task
 export CUDA_VISIBLE_DEVICES=$1
 nohup python -u -m main \
 --mode=train \
 --batch_size=64 \
 --epochs=30 \
 --early_stop=0 \
 --model_name='figure_'$2'_'$3 \
 --model=$2 \
 --task=$4 \
 --slm_weight=$3 \
 --new_model > logs/'figure_'$2'_'$3 2>&1&
  # --pre_dataset \
	#	--multi_domain \

