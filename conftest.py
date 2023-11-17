#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import os
from datetime import timedelta

from hypothesis import settings

settings.register_profile("ci", max_examples=1000, deadline=timedelta(seconds=1))
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))
