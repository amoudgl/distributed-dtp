import torch


class VanillaTrainer:
    """
    Single GPU vanilla trainer mimicking pytorch lightning trainer with limited support.

    Works with vanilla baseline model, can be tested with the following command:
    python main.py model=vanilla_baseline trainer=vanilla scheduler=cosine network=simple_vgg datamodule=cifar10
    """

    def __init__(self, gpus, max_epochs, logger) -> None:
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.logger = logger
        self.device = "cpu" if not torch.cuda.is_available() else torch.cuda.current_device()

    def fit(self, model, datamodule):
        # init
        datamodule.setup(stage="fit")
        optim_config = model.configure_optimizers()
        lr_scheduler = getattr(optim_config, "lr_scheduler", None)
        model = model.to(self.device)
        model.trainer = self  # set trainer as model's attribute like lightning

        # training loop
        for epoch in range(self.max_epochs):
            # run training epoch
            self.train_epoch(model, datamodule.train_dataloader(), optim_config)

            # evaluate model on validation set
            metric = self.val_epoch(model, datamodule.val_dataloader())

            if (
                lr_scheduler is not None
                and lr_scheduler["interval"] == "epoch"
                and epoch % lr_scheduler["frequency"] == 0
            ):
                scheduler = lr_scheduler["scheduler"]
                scheduler.step()

            print(f"epoch: {epoch}, val metric: {metric}")

    def train_epoch(self, model, train_dataloader, optim_config):
        model.train()
        optimizer = optim_config["optimizer"]
        lr_scheduler = getattr(optim_config, "lr_scheduler", None)
        losses = []

        for step, batch in enumerate(train_dataloader):
            # transfer batch to device
            batch = tuple(t.to(device=self.device) for t in batch)

            # forward pass
            loss = model.training_step(batch, step)

            # backward pass + update
            if model.automatic_optimization:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # scheduler step
            if (
                lr_scheduler is not None
                and lr_scheduler["interval"] == "step"
                and step % lr_scheduler["frequency"] == 0
            ):
                scheduler = lr_scheduler["scheduler"]
                scheduler.step()
            losses.append(loss.item())
        return torch.tensor(losses).mean()

    def val_epoch(self, model, val_dataloader):
        model.eval()
        metrics = []

        for step, batch in enumerate(val_dataloader):
            # transfer batch to device
            batch = tuple(t.to(device=self.device) for t in batch)

            # forward pass
            metric = model.validation_step(batch, step)
            metrics.append(metric.item())

        return torch.tensor(metrics).mean()

    def test(self, model, datamodule, verbose=False):
        # verbose argument is just a dummy argument to match lightning format
        datamodule.setup(stage="test")
        return self.val_epoch(model, datamodule.test_dataloader())
