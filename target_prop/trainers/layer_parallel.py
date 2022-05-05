import torch


class LayerParallelTrainer:
    """
    Multi-GPU layer parallel trainer for DTP.

    Works with DTP layer parallel model, can be tested with the following command:
    python main.py model=layer_parallel_dtp trainer=layer_parallel scheduler=cosine network=simple_vgg datamodule=cifar10
    """

    def __init__(self, gpus, max_epochs, logger) -> None:
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.logger = logger
        self.device = "cpu" if not torch.cuda.is_available() else torch.cuda.current_device()
        pass

    def fit(self, model, datamodule, rank=0):
        pass
        # # sync params across processes
        # model.train()

        # # training loop
        # for epoch in range(self.max_epochs):
        #     # in each batch
        #     # do feedback training according to rank
        #     for batch in

        #     # all_gather feedback params or just gather

        #     # forward + backward step

        #     # (optional) broadcast params to other devices
