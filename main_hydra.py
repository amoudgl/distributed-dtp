import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # print whole config
    print(OmegaConf.to_yaml(cfg))

    # create datamodule
    datamodule = instantiate(cfg.datamodule,
                             data_dir=cfg.data_dir)

    # create network
    network: nn.Sequential = instantiate(cfg.network,
                                         in_channels=datamodule.dims[0],
                                         n_classes=datamodule.num_classes)

    # create model
    model = instantiate(cfg.model,
                        _recursive_=False,  # do not create schedulers by default
                        datamodule=datamodule,
                        network=network,
                        hparams=cfg.model,
                        full_config=cfg,
                        network_hparams=cfg.network)

    # create trainer
    if cfg.debug:
        # define trainer debug behavior
        cfg.trainer.accelerator = None
        cfg.trainer.gpus = 1
        cfg.trainer.logger = None  # disable wandb logging
        cfg.trainer.enable_checkpointing = False
        cfg.trainer.profiler = "simple"
    trainer = instantiate(cfg.trainer)

    # run experiment
    trainer.fit(model, datamodule=datamodule)

    # run on test set
    test_results = trainer.test(model, datamodule=datamodule, verbose=True)

    # display experiment results
    wandb.finish()
    top1_accuracy: float = test_results[0]["test/accuracy"]
    top5_accuracy: float = test_results[0]["test/top5_accuracy"]
    print(f"Test top1 accuracy: {top1_accuracy:.1%}")
    print(f"Test top5 accuracy: {top5_accuracy:.1%}")
    return top1_accuracy, top5_accuracy

if __name__ == "__main__":
    main()
    