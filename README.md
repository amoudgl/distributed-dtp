# Distributed DTP

This repository contains the distributed implementation of our work "Towards Scaling Difference Target Propagation with Backprop Targets" accepted to ICML 2022. It implements a custom distributed scheme for our proposed Difference Target Propagation (DTP) algorithm by parallelizing feedback weight training.

![implementation design](img/distributed_dtp_design.png)

## Setup

Install python packages.
```
pip install -r requirements.txt
```

Download required datasets [[1](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), [2](https://drive.google.com/uc?id=1XAlD_wshHhGNzaqy8ML-Jk0ZhAm8J5J_)] and unzip them in `data/` directory.

The following code runs on Python > 3.6 with Pytorch 1.7.0.

## Implementation Notes

- This codebase uses [Hydra](https://hydra.cc/) which builds config on the fly as per user commands and defaults specified in `conf/config.yaml`. However, we also provide full config files in `conf/reproduce` directory for reproducing all the experiments and they can be launched directly from yaml config file as follows:
    ```
    python main.py --config-path conf/reproduce --config-name cifar10_simple_vgg_dtp
    ```
- Since objects are instantiated directly from config, Hydra configuration in this codebase _strictly_ follows class hierarchy.
- The distributed trainer follows [pytorch-lightning](https://www.pytorchlightning.ai/) API and model implementation is completely separate from the trainer. This enables plug-and-play with different models using the same trainer.
- This implementation uses git pre-commit hooks provided in `.pre-commit-config.yaml` for consistent formatting across the whole codebase.

## Running the code


### Distributed implementation

For fast DTP training with layer parallelism on CIFAR-10, do:
```
python main.py model=layer_parallel_dtp \
    trainer=layer_parallel \
    scheduler=cosine \
    network=simple_vgg \
    datamodule=cifar10 \
    datamodule.num_workers=1 \
    trainer.gpus=6
```
The above command will launch N processes corresponding to N layers in the backward net. Any number of GPUs can be passed via command line; each process will use GPU device `rank % trainer.gpus`.


It is recommended to use `trainer.gpus=1` for debugging.


For ImageNet, do:
```
python main.py model=layer_parallel_dtp \
    trainer=layer_parallel \
    scheduler=cosine \
    network=simple_vgg \
    datamodule=imagenet32 \
    datamodule.num_workers=1 \
    trainer.gpus=6 \
    datamodule.batch_size=256 \
    model.hparams.feedback_training_iterations=[25,35,40,60,25] \
    model.hparams.f_optim.lr=0.01 \
    model.hparams.b_optim.momentum=0.9 \
    model.hparams.b_optim.lr=[1e-4,3.5e-4,8e-3,8e-3,0.18]
```

### Sequential implementation
To train with DTP on CIFAR-10, do:
```
python main.py \
    model=dtp \
    network=simple_vgg \
    datamodule=cifar10 \
    trainer=default \
    scheduler=cosine
```

The following example demonstrates overriding the default config through command line:
```
python main.py \
    model=dtp \
    network=simple_vgg \
    datamodule=imagenet32 \
    trainer=default \
    scheduler=cosine \
    seed=123 \
    datamodule.batch_size=256 \
    model.hparams.feedback_training_iterations=[25,35,40,60,25] \
    model.hparams.f_optim.lr=0.01 \
    model.hparams.b_optim.momentum=0.9 \
    model.hparams.b_optim.lr=[1e-4,3.5e-4,8e-3,8e-3,0.18]
```

### Reproducibility

All the experiments can be reproduced from config files provided in `conf/reproduce` directory like below:
```
python main.py --config-path conf/reproduce --config-name <name>
```

NOTE: `data_dir` field in above config files should be modified as per your setup before launching experiments.

## Results

This distributed implementation achieves the same overall performance as the default sequential one but finishes much faster depending on the resources available for workers. Following is an example comparing the two implementations on CIFAR-10:

![results](img/training_plots.png)

(Left) Top1 validation accuracy plots on CIFAR-10 comparing default sequential DTP and our distributed DTP implementation. Our distributed DTP implementation achieves same performance as sequential DTP which uses 1 RTX 3090 GPU. (Right) Same Top1 validation accuracy on CIFAR-10 plotted against wall clock time. Our distributed DTP implementation finishes in less than 1/4th time of the sequential DTP by utilizing 2 RTX 3090 GPUs.

## License
MIT
