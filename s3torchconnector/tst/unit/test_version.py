#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from s3torchconnector import __version__


def test_connector_version():
    assert isinstance(__version__, str)
    assert __version__ > "1.0.0"
