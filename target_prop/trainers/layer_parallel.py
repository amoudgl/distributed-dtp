import torch


class LayerParallelTrainer:
    def __init__(self, gpus, max_epochs, logger) -> None:
        # self.max_epochs = max_epochs
        # self.gpus = gpus
        # self.logger = logger
        # setup DDP processes
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

        #     # all_gather params or just gather

        #     # forward + backward step

        #     # (optional) broadcast params to other devices
