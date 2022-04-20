from dataclasses import dataclass
from logging import getLogger
from typing import List, Union

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from simple_parsing.helpers import list_field
from simple_parsing.helpers.hparams import log_uniform, uniform
from torch import Tensor, nn

from .vanilla_dtp import VanillaDTP

logger = getLogger(__name__)


class TargetProp(VanillaDTP):
    """Target Propagation (TP)."""

    def __init__(
        self,
        datamodule: LightningDataModule,
        network: nn.Sequential,
        hparams: DictConfig,
        config: DictConfig,
        network_hparams: DictConfig,
    ):
        super().__init__(datamodule, network, hparams, config, network_hparams)

    def compute_target(self, i: int, G: nn.Module, hs: List[Tensor], prev_target: Tensor) -> Tensor:
        """Compute the target of the previous forward layer. given ,
        the associated feedback layer, the activations for each layer, and the target of the current
        layer.

        Parameters
        ----------
        i : int
            the index of the forward layer for which we want to compute a target
        G : nn.Module
            the associated feedback layer
        hs : List[Tensor]
            the activations for each layer
        prev_target : Tensor
            The target of the next forward layer.

        Returns
        -------
        Tensor
            The target to use to train the forward layer at index `i`.
        """
        # NOTE: Target propagation:
        return G(prev_target)
        # NOTE: Difference target propagation (both Vanilla and DTP-J):
        # return hs[i - 1] - G(hs[i]) + G(prev_target)
        # Cooler ordering, from the Meuleman's DTP paper:
        # return G(prev_target) + (hs[i - 1] - G(hs[i]))

    def layer_feedback_loss(
        self,
        *,
        feedback_layer: nn.Module,
        forward_layer: nn.Module,
        input: Tensor,
        output: Tensor,
        noise_scale: Union[float, Tensor],
        noise_samples: int = 1,
    ) -> Tensor:
        # NOTE: The feedback loss in Target Propagation is the same as in (Vanilla)
        # Difference Target Propagation (as far as I can tell.)
        # TODO: Confirm this with @ernoult.
        return super().layer_feedback_loss(
            feedback_layer=feedback_layer,
            forward_layer=forward_layer,
            input=input,
            output=output,
            noise_scale=noise_scale,
            noise_samples=noise_samples,
        )
