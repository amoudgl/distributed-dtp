import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pytorch_lightning.utilities.seed import seed_everything
from torch import Tensor
from tqdm import tqdm


def init_process(rank, size, fn, backend, *args):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "9000"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, *args)


class LayerParallelTrainer:
    """
    Multi-GPU layer parallel trainer for DTP.
    """

    def __init__(self, gpus, max_epochs, seed, backend="gloo") -> None:
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.seed = seed
        self.backend = backend

    def fit(self, model, datamodule):
        # setup distributed processes
        processes = []
        mp.set_start_method("spawn")
        # number of layers must be equal to number of processes for layer parallel feedback weight training
        size = len(model.backward_net)
        for rank in range(size):
            # set different GPU for each process
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % self.gpus)
            p = mp.Process(
                target=init_process,
                args=(rank, size, self.fit_worker, self.backend, model, datamodule),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def fit_worker(self, rank, size, model, datamodule):
        self.device = torch.device("cuda:0")  # each process will see only one GPU
        print(f"[rank {rank}] using seed: {self.seed}")

        # we set same seed for each process since we want to have exact same
        # batch on every process, we just parallelize the feedback training not data
        seed_everything(self.seed, workers=True)

        datamodule.setup(stage="fit")
        dist.barrier()  # wait for dataloading on all processes to finish
        # test if you get same batches
        # batch = next(iter(datamodule.train_dataloader()))
        # print(f"rank {rank}, seed {self.seed}, batch labels: {batch[1]}")

        optim_config = model.configure_optimizers()
        self.setup_optim(optim_config)
        scheduler = self._schedulers[0] if len(self._schedulers) > 0 else None
        model = model.to(self.device)
        model.trainer = self  # set trainer as model's attribute like lightning

        # broadcast params from rank 0 to all other processes just to be safe
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

        # training loop
        for epoch in range(self.max_epochs):
            # run training epoch
            self.train_epoch(model, datamodule.train_dataloader(), optim_config)

            # evaluate model on validation set
            top1, top5 = self.val_epoch(model, datamodule.val_dataloader())

            # scheduler step
            if scheduler:
                scheduler.step()
            if dist.get_rank() == 0:
                summary = f"[epoch {epoch}] top1:{top1:4.4f} top5:{top5:4.4f}"
                print(summary)

    def train_epoch(self, model, train_dataloader, optim_config):
        model.train()
        model.optimizers = self.optimizers
        model.lr_schedulers = self.lr_schedulers
        losses = []
        rank = dist.get_rank()
        if rank == 0:
            pbar = tqdm(total=len(train_dataloader), smoothing=0.0)

        for step, batch in enumerate(train_dataloader):
            # transfer batch to device
            batch = tuple(t.to(device=self.device) for t in batch)

            # feedback weight training for a layer corresponding to rank
            x, y = batch
            feedback_training_outputs: Dict = model.feedback_loss(x, y, rank=rank, phase="train")

            # sync feedback net params
            updated_param_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(updated_param_list, model.backward_net[::-1][rank].state_dict())
            for i, layer in enumerate(model.backward_net[::-1]):
                layer.load_state_dict(updated_param_list[i])

            # broadcast rng state so that all processes do same forward update
            rng_state = torch.cuda.get_rng_state()
            dist.broadcast(rng_state, src=0)
            torch.cuda.set_rng_state(rng_state)

            # target propagation and forward update
            forward_training_outputs: Dict = model.forward_loss(x, y, phase="train")
            forward_loss: Tensor = forward_training_outputs["loss"]
            forward_optimizer = model.forward_optimizer
            forward_optimizer.zero_grad()
            forward_loss.backward()
            forward_optimizer.step()
            forward_loss = forward_loss.detach()
            last_layer_loss: Tensor = forward_training_outputs["layer_losses"][-1].detach()
            if rank == 0:
                pbar.set_description(
                    "loss: {:4.4f}, top1: {:4.4f}".format(
                        last_layer_loss.item(), forward_training_outputs["top1_acc"].item()
                    )
                )
                pbar.update(1)
            losses.append(last_layer_loss.item())
        return torch.tensor(losses).mean()

    def setup_optim(self, optim_config):
        self._optimizers = []
        self._schedulers = []
        for config in optim_config:
            optimizer = config["optimizer"]
            self._optimizers.append(optimizer)
            if "lr_scheduler" in config:
                scheduler = config["lr_scheduler"]["scheduler"]
                self._schedulers.append(scheduler)

    def optimizers(self):
        """lightning-like method to get optimizers."""
        return self._optimizers

    def lr_schedulers(self):
        return self._schedulers

    def val_epoch(self, model, val_dataloader):
        model.eval()
        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print("evaluating...")
            pbar = tqdm(total=len(val_dataloader), smoothing=0.0)

        for step, batch in enumerate(val_dataloader):
            # transfer batch to device
            batch = tuple(t.to(device=self.device) for t in batch)

            # forward pass
            x, y = batch
            output: Dict = model.forward_loss(x, y, phase="val")

            # update metrics
            top1.update(output["top1_acc"], x.size(0))
            top5.update(output["top5_acc"], x.size(0))
            if rank == 0:
                pbar.update(1)

        return top1.avg, top5.avg

    def test(self, model, datamodule, verbose=False):
        # verbose argument is just a dummy argument to match lightning format
        datamodule.setup(stage="test")
        if not hasattr(self, "device"):
            # use current device when model is directly tested without training
            self.device = torch.cuda.current_device()
            model = model.to(self.device)
        top1, top5 = self.val_epoch(model, datamodule.test_dataloader())
        # keep return format for test method consistent with other trainers
        return [{"test/accuracy": top1, "test/top5_accuracy": top5}]


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """
    Computes and stores the average and current value
    (Cloned from PyTorch examples repo)
    """

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)
