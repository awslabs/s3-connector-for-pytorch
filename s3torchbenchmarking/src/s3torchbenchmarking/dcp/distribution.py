#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import TypedDict

import numpy as np


class Statistics(TypedDict, total=False):
    n: int  # number of elements in the distribution
    unit: str
    value: float  # when the number of elements is one, `value` will contain it
    mean: float
    min: float
    max: float


class Distribution(list):
    """Subclass of :obj:`list`, adding statistics dump (min, max, etc.)."""

    def dump(self, unit: str) -> Statistics:
        if not self:
            return {}
        if len(self) == 1:
            return {
                "n": 1,
                "unit": unit,
                "value": self[0],
            }
        return {
            "n": len(self),
            "unit": unit,
            "mean": np.nanmean(self),  # `nan*` methods ignore NaN values, just in case
            "min": np.nanmin(self),
            "max": np.nanmax(self),
        }
