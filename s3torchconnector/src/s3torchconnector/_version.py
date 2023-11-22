#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#   // SPDX-License-Identifier: BSD

import importlib.metadata

# __package__ is 's3torchconnector'
__version__ = importlib.metadata.version(__package__)
user_agent_prefix = f"{__package__}/{__version__}"
