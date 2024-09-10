export PYTHONPATH=./

export CUDA_VISIBLE_DEVICES=3

slakh_train_metadata_path='./data/metadata/slakh_train_fs1_hs0.5_augmented_as5_active_chunks.csv'
slakh_validation_metadata_path='./data/metadata/slakh_validation_fs5_hs1_not_augmented_as0_active_chunks.csv'
config_path="./configs.json"
log_root="./logfiles"
CUDA_LAUNCH_BLOCKING=1

echo "Train SourceVAE"
python3 train.py \
    --slakh_train_metadata_path ${slakh_train_metadata_path}\
    --slakh_validation_metadata_path ${slakh_validation_metadata_path}\
    --config ${config_path} \
    --checkpoint_path ${log_root} \
    --checkpoint_interval 5000 \
    --summary_interval 5 \
    --validation_interval 5000 \
    --training_epochs 10000 \
    --stdout_interval 5 \