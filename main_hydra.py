import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # print whole config
    print(OmegaConf.to_yaml(cfg))

    # create datamodule
    datamodule = instantiate(cfg.datamodule)

    # # create network
    print(datamodule.dims[0])
    # print(datamodule.num_classes)
    network: nn.Sequential = instantiate(cfg.network,
                                         in_channels=datamodule.dims[0],
                                         n_classes=datamodule.num_classes)
    print(network)

if __name__ == "__main__":
    my_app()
    