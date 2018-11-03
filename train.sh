export CUDA_VISIBLE_DEVICES=$1
python -u -m main \
--epochs=10 \
--new_model=True \
--model_name=s2s \
--model=s2s \
--task=kvret

