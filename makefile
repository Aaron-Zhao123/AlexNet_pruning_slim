DATA_DIR=/local/scratch/ssd
TRAIN_DIR=/local/scratch/yaz21/tmp
run: alexnet_train.py
	python alexnet_train.py --num_gpus=2 --batch_size=256 --train_dir=$(TRAIN_DIR) --data_dir=$(DATA_DIR)

git-add:
	git add -A
	git commit -m"editing"
	git push

git-fetch:
	git fetch
	git merge
