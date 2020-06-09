# KDD Cup 2020 AutoGraph Challenge

8th place solution.

Competition page:

https://www.automl.ai/competitions/3

https://www.4paradigm.com/competition/kddcup2020


# Usage

Class `Model` from `model.py` implements the API required by the evaluation system.

Example of local running:
```bash
docker run --gpus=0 --shm-size=30G -it --rm -v "$(pwd):/app/autograph" -v /tmp/pipdocker:/root/.cache/pip -w /app/autograph nehzux/kddcup2020:v2
```

```
python starting_kit/run_local_test.py --dataset_dir=./starting_kit/data/demo/ --code_dir=./src/
```

Please refer to the [official documentation](https://www.automl.ai/competitions/3#learn_the_details-credits) for the detailed interface description.


# How it works

Let's just train several different architectures for the node classification task:

* TagConv - [Topology Adaptive Graph Convolutional Networks](https://arxiv.org/pdf/1710.10370.pdf)
* SageConv - [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf)
* GraphConv - [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
* SGConv - [Simplifying Graph Convolutional Networks](https://arxiv.org/pdf/1902.07153.pdf)

And then average the results of the top performing models (evaluated on the validation).

# Acknowledgements

* [DGL Library](https://github.com/dmlc/dgl)
* [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric)
* [Ray.Tune](https://docs.ray.io/en/latest/tune.html)
