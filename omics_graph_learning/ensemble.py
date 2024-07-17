#! /usr/bin/env python
# -*- coding: utf-8 -*-
#


"""Module to _ensemble models multiple models. On finalized models, validation is
split into K-folds. K # of models are trained on the training set plus one
K-validation fold. Models are then averaged to create the finalized model."""


from pathlib import Path
from typing import List

import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    """Class to handle ensembling of multiple trained models. Ensembling happens
    during object instantiation.

    Attributes:
        model_paths: List of paths to models to ensemble.

    Examples:
    --------
    >>> pass
    >>> pass
    """

    def __init__(
        self,
        model_paths: List[Path],
    ):
        super(EnsembleModel, self).__init__()
        self.model_paths = model_paths

        # load models and _ensemble
        models = self._get_models()
        self.ensembled_model = self._ensemble(models)

    def _get_models(self) -> List[nn.Module]:
        """Ensemble state dicts of models"""
        return [torch.load(model_path).state_dict() for model_path in self.models_paths]

    def _ensemble(self, models: List[nn.Module]) -> nn.Module:
        """Ensemble models"""
        ensembled_states = {
            key: torch.stack([model.state_dict()[key] for model in models]).mean(0)
            for key in models[0].state_dict().keys()
        }

        # create new nn.Module instance
        ensembled_model = type(models[0])()
        ensembled_model.load_state_dict(ensembled_states)
        return ensembled_model
