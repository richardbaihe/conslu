export CUDA_VISIBLE_DEVICES=$1
python -u -m main \
--pre_dataset=False \
--epochs=20 \
--new_model=True \
--model_name=sden \
--model=sden \
--task=kvret \
--slm_weight=0
