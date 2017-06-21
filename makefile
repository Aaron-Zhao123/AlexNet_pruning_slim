DATA_DIR=/local/scratch/ssd
TRAIN_DIR=/local/scratch/yaz21/tmp
python alexnet_train.py --num_gpus=2 --batch_size=32 --train_dir=$TRAIN_DIR --data_dir=$DATA_DIR
