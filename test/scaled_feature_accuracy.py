#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Ensure parity of node features post-scaling. This class is used within the
script that scales node features, so the tests are done during runtime."""


import logging
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import RobustScaler  # type: ignore


class ScaledFeatureAccuracy:
    """Class meant to organize tests that check the accuracy of node features
    post-scaling.
    """

    def __init__(
        self,
        original: np.ndarray,
        scaled: np.ndarray,
        split: Dict[str, List[str]],
        idxs: Dict[str, int],
        continous_feat_range: int,
        logger: logging.Logger,
        scalers: Dict[int, RobustScaler],
    ) -> None:
        """Instantiate the NodeFeatureAccuracy class."""
        self.original = original
        self.scaled = scaled
        self.split = split
        self.idxs = idxs
        self.scalers = scalers
        self.logger = logger

        self.feat_range = continous_feat_range
        self.test_idxs = [idxs[gene] for gene in split["test"] if gene in idxs]
        self.val_idxs = [idxs[gene] for gene in split["validation"] if gene in idxs]

    def test_feature_order(self) -> bool:
        """Check that the original and scaled have their node features in the same order."""
        for i in range(self.feat_range):
            original_col = self.original[:, i]
            scaled_col = self.scaled[:, i]

            original_order = np.argsort(original_col)
            scaled_order = np.argsort(scaled_col)

            if not np.array_equal(original_order, scaled_order):
                self.logger.warning(f"Order mismatch detected in feature {i}")
                return False
        return True

    def test_feature_shape(self) -> None:
        """Check that the original and scaled node features have the same shape.
        Additionally, check that one-hot encoded features do not change."""
        assert (
            self.original.shape == self.scaled.shape
        ), "Shapes don't match after scaling!"
        assert np.array_equal(
            self.original[:, self.feat_range :], self.scaled[:, self.feat_range :]
        ), "One-hot encoded features changed unexpectedly!"
        assert not np.array_equal(
            self.original[:, : self.feat_range], self.scaled[:, : self.feat_range]
        ), "Continuous features did not scale!"

    def test_nan_inf(
        self,
    ) -> None:
        """Check for NaNs and Infs in the scaled node features"""
        assert not np.isnan(self.scaled).any(), "NaN values found in scaled features!"
        assert not np.isinf(self.scaled).any(), "Inf values found in scaled features!"

        assert not np.isnan(
            self.scaled[self.test_idxs]
        ).any(), "NaN values found in test set!"
        assert not np.isinf(
            self.scaled[self.test_idxs]
        ).any(), "Inf values found in test set!"
        assert not np.isnan(
            self.scaled[self.val_idxs]
        ).any(), "NaN values found in validation set!"
        assert not np.isinf(
            self.scaled[self.val_idxs]
        ).any(), "Inf values found in validation set!"

    def manual_spotcheck(self, n_tests: int = 1000) -> bool:
        """Randomly select n_tests and spot check node features. Manually scales
        a random point and compares to the scaled point.
        """
        n_features = self.original.shape[1] - self.feat_range
        n_tests = min(n_tests, n_features * self.original.shape[0])

        # randomly select features and samples
        feature_indices = np.random.randint(0, self.feat_range, n_tests)
        sample_indices = np.random.randint(0, self.original.shape[0], n_tests)

        for i in range(n_tests):
            feat_idx = feature_indices[i]
            sample_idx = sample_indices[i]

            original_value = self.original[sample_idx, feat_idx]
            scaled_value = self.scaled[sample_idx, feat_idx]

            # manually calculate the expected scaled value
            scaler = self.scalers[feat_idx]
            expected_scaled_value = scaler.transform([[original_value]])[0][0]

            # compare the values
            if not np.isclose(scaled_value, expected_scaled_value):
                self.logger.error(
                    f"Mismatch at feature {feat_idx}, sample {sample_idx}: "
                    f"Expected {expected_scaled_value}, got {scaled_value}"
                )
                return False

        self.logger.info(f"All {n_tests} randomly checked features passed the test.")
        return True

    def run_all_tests(
        self,
    ) -> bool:
        """Run all tests and return True if all pass, False otherwise."""
        try:
            if not self.test_feature_order():
                self.logger.error("Feature order not preserved after scaling!")
                return False

            self.test_feature_shape()
            self.test_nan_inf()

            if not self.manual_spotcheck():
                self.logger.error("Random feature check failed!")
                return False

            return True
        except AssertionError as e:
            self.logger.error(f"Test failed: {str(e)}")
            return False
