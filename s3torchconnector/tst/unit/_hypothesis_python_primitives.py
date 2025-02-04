#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from hypothesis.strategies import (
    integers,
    characters,
    floats,
    booleans,
    deferred,
    tuples,
    dictionaries,
    lists,
    text,
)

scalars = (
    booleans()
    | integers()
    # Disallow nan as it doesn't have self-equality
    | floats(allow_nan=False)
    | characters()
    | text(max_size=10)
)

hashable = deferred(
    lambda: (scalars | tuples(hashable))
)

python_primitives = deferred(
    lambda: (
        hashable
        | lists(python_primitives, max_size=5)
        | dictionaries(keys=hashable, values=python_primitives, max_size=3)
    )
)
