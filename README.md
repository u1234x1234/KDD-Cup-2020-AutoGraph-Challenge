# autograph2020

docker run --gpus=0 --shm-size=30G -it --rm -v "$(pwd):/app/autograph" -v /tmp/pipdocker:/root/.cache/pip -w /app/autograph nehzux/kddcup2020:v2


python starting_kit/run_local_test.py --dataset_dir=./starting_kit/data/demo/ --code_dir=./src/
