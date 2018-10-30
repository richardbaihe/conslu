export CUDA_VISIBLE_DEVICES=$1
python -u -m main \
--epochs=10 \
--model_name=context_s2s \
--model=context_s2s
