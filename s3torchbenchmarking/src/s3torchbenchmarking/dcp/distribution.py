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
    p50: float
    p75: float
    p90: float
    max: float


class Distribution(list):
    """Extension of :obj:`list`, adding statistics dump (min, max, median, etc.)."""

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
            "mean": np.nanmean(self),  # `nan*` methods ignore NaN values
            "min": np.nanmin(self),
            "p50": np.nanmedian(self),
            "p75": np.nanpercentile(self, 75),
            "p90": np.nanpercentile(self, 90),
            "max": np.nanmax(self),
        }
