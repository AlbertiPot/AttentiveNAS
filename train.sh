# nohup \
# python -u train_attentive_nas.py \
# --config-file configs/train_attentive_nas_models.yml \
# --machine-rank 0 \
# --num-machines 1 \
# > train.log 2>&1 &

# dali
nohup \
python -u dali_train.py \
--config-file configs/train_attentive_nas_models.yml \
--machine-rank 0 \
--num-machines 1 \
> train_test.log 2>&1 &
