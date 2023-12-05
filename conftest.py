#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import os
from datetime import timedelta

import pytest
from hypothesis import settings

settings.register_profile("ci", max_examples=1000, deadline=timedelta(seconds=1))
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))

per_test_timeout = int(os.getenv("TEST_TIMEOUT", "120"))


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(per_test_timeout))
