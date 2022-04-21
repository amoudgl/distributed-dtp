# Revisiting Difference Target Propagation

The following code runs on Python > 3.6 with Pytorch 1.7.0.
## Installation
```console
pip install -e .
```

## Running the code (Hydra)

To train with DTP on CIFAR-10, do:
```
python main_hydra.py \
model=dtp \
network=simple_vgg \
datamodule=cifar10 \
trainer=default \
scheduler=cosine
```

To reproduce experiment results from a complete config, just do:
```
python main_hydra.py reproduce=imagenet32_simple_vgg_dtp
```

Following example demonstrates overriding config through command line:
```
python main_hydra.py \
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

### Legacy Implementation
To check training on CIFAR-10, type the following command in the terminal:

```console
python main.py --batch-size 128 \
    --C 128 128 256 256 512 \
    --iter 20 30 35 55 20 \
    --epochs 90 \
    --lr_b 1e-4 3.5e-4 8e-3 8e-3 0.18 \
    --noise 0.4 0.4 0.2 0.2 0.08 \
    --lr_f 0.08 \
    --beta 0.7 \
    --path CIFAR-10 \
    --scheduler --wdecay 1e-4
```
