# Revisiting Difference Target Propagation

The following code runs on Python > 3.6 with Pytorch 1.7.0.
## Installation
```console
pip install -e .
```


## Running the code (Hydra)

```
python main_pl.py \
model=dtp \
network=simple_vgg \
dataset=imagenet32 \
trainer=default \
seed=123 \
dataset.batch_size=256 \
model.f_optim.type=sgd \
model.f_optim.lr=0.01 \
model.b_optim.feedback_training_iterations=[25,35,40,60,25] \
model.b_optim.type=sgd \
model.b_optim.momentum=0.9 \
model.b_optim.lr=[1e-4,3.5e-4,8e-3,8e-3,0.18] \
model.scheduler=cosine \
```


## Running the code
To run the pytorch-lightning re-implementation of DTP on CIFAR-10, use the following command:
```console
python main_pl.py run dtp simple_vgg
```

To use the modified version of the above DTP model, with "parallel" feedback weight training on CIFAR-10, use the following command:
```console
python main_pl.py run parallel_dtp simple_vgg
```

### ImageNet

To train with DTP on ImageNet 32x32 dataset, do:
```
python main_pl.py run dtp simple_vgg \
--batch_size 256 \
--num_workers 4 \
--dataset imagenet32 \
--seed 123 \
--f_optim.type sgd \
--f_optim.lr 0.01 \
--feedback_training_iterations 25 35 40 60 25 \
--b_optim.type sgd \
--b_optim.momentum 0.9 \
--b_optim.lr 1e-4 3.5e-4 8e-3 8e-3 0.18 \
--use_scheduler true \
cosine
```

To train with backprop baseline on ImageNet 32x32 dataset, do:
```
python main_pl.py run backprop simple_vgg \
--batch_size 256 \
--num_workers 4 \
--dataset imagenet32 \
--seed 123 \
--type sgd \
--lr 0.01 \
--use_scheduler true \
step \
--step_size 30
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
