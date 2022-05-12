import gc
import os
import time

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm


@hydra.main(config_path="../conf", config_name="config")
def run(cfg: DictConfig):
    def memory_usage():
        torch.cuda.synchronize()
        used_memory = torch.cuda.max_memory_allocated()
        return used_memory

    hist_accuracies = []
    hist_durations = []
    hist_memory = []
    torch.backends.cudnn.deterministic = True
    for i in tqdm(
        range(cfg.num_runs), desc=f"Benchmarking {cfg.model._target_} with {cfg.trainer._target_}"
    ):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()
        time.sleep(1)

        # create datamodule
        datamodule = instantiate(cfg.datamodule, data_dir=cfg.data_dir)

        # create network
        network: nn.Sequential = instantiate(
            cfg.network, in_channels=datamodule.dims[0], n_classes=datamodule.num_classes
        )

        # create model
        # note: _recursive_ is set to False below to avoid creating optimizers
        # since they are created internally in configure_optimizers() method
        # of lightning module.
        model = instantiate(
            cfg.model,
            _recursive_=False,
            datamodule=datamodule,
            network=network,
            hparams=cfg.model,
            full_config=cfg,
            network_hparams=cfg.network,
        )

        # create trainer for benchmarking
        cfg.trainer.max_epochs = 10
        trainer = instantiate(cfg.trainer)

        # run experiment
        time_start = time.perf_counter()
        trainer.fit(model, datamodule=datamodule)
        used_memory = memory_usage()
        time_end = time.perf_counter()

        # run on test set
        test_results = trainer.test(model, datamodule=datamodule, verbose=True)
        top1_accuracy: float = test_results[0]["test/accuracy"]
        top5_accuracy: float = test_results[0]["test/top5_accuracy"]

        # bookkeeping
        hist_durations.append(time_end - time_start)
        hist_accuracies.append([top1_accuracy, top5_accuracy])
        hist_memory.append(used_memory)

    print("{cfg.num_runs} runs benchmark results:")
    print("durations: {hist_durations}")
    print("accuracies ([top1, top5]): {hist_accuracies}")
    print("memory: {hist_memory}")


if __name__ == "__main__":
    run()
