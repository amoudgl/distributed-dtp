## Benchmark

You can use the benchmark script to log time, memory usage and performance of any run. It keeps the same training config but just trains for 10 epochs N times where N can be passed through command line by `+num_runs=10`. Following are some commands useful for benchmarking:


Sequential DTP
```
python benchmark.py \
    model=dtp \
    network=simple_vgg \
    datamodule=cifar10 \
    trainer=default \
    scheduler=cosine \
    trainer.logger=false \
    ++num_runs=10 \
    ++trainer.enable_checkpointing=false \
    ++trainer.limit_val_batches=0 \
    ++trainer.num_sanity_val_steps=0 \
    ++trainer.enable_model_summary=false

```


Layer parallel DTP
```
python benchmark.py \
    model=layer_parallel_dtp \
    trainer=layer_parallel \
    scheduler=cosine \
    network=simple_vgg \
    datamodule=cifar10 \
    datamodule.num_workers=1 \
    trainer.gpus=6 \
    ++num_runs=10
```
