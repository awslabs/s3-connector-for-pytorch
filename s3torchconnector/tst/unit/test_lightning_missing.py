#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from importlib.util import find_spec

import pytest


#  Skip if lightning is installed
@pytest.mark.skipif(
    find_spec("lightning"),
    reason="Test verifies error message if lightning extension is used without installation",
)
def test_lightning_not_installed():
    with pytest.raises(ModuleNotFoundError) as e:
        import s3torchconnector.lightning
    assert str(e.value) == "No module named 'lightning'"
