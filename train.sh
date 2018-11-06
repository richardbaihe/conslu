export CUDA_VISIBLE_DEVICES=$1
python -u -m main \
--mode=eval \
--epochs=30 \
--early_stop=7 \
--model_name=sden_slm \
--model=sden \
--task=kvret \
--slm_weight=0.3 \
#--new_model
