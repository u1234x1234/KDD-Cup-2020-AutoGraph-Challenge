# autograph2020

docker run --gpus=0 -it --rm -v "$(pwd):/app/autograph" -w /app/autograph nehzux/kddcup2020:v2


python run_local_test.py --dataset_dir=./data/demo/ --code_dir=../ag/
