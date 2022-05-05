import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, size, fn, backend, *args):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "9000"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, *args)


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

    def fit(self, model, datamodule):
        # setup distributed processes
        processes = []
        mp.set_start_method("spawn")
        size = self.gpus
        for rank in range(size):
            p = mp.Process(
                target=init_process, args=(rank, size, self.fit_worker, "nccl", model, datamodule)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        # broadcast params from rank 0 to all other processes
        # model.train()

        # # training loop
        # for epoch in range(self.max_epochs):
        #     # in each batch
        #     # do feedback training according to rank
        #     for batch in

        #     # all_gather feedback params or just gather

        #     # forward + backward step

        #     # (optional) broadcast params to other devices

    def fit_worker(self, rank, size, model, datamodule):
        print(
            rank,
        )

    def test(self, model, datamodule, verbose=False):
        # verbose argument is just a dummy argument to match lightning format
        pass
