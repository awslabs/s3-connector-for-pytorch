#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from hypothesis.strategies import (
    integers,
    binary,
    none,
    characters,
    complex_numbers,
    floats,
    booleans,
    decimals,
    fractions,
    deferred,
    frozensets,
    tuples,
    dictionaries,
    lists,
    uuids,
    sets,
    text,
)

scalars = (
    none()
    | booleans()
    | integers()
    # Disallow nan as it doesn't have self-equality
    | floats(allow_nan=False)
    | complex_numbers(allow_nan=False)
    | decimals(allow_nan=False)
    | fractions()
    | characters()
    | binary(max_size=10)
    | text(max_size=10)
    | uuids()
)

hashable = deferred(
    lambda: (scalars | frozensets(hashable, max_size=5) | tuples(hashable))
)

python_primitives = deferred(
    lambda: (
        hashable
        | sets(hashable, max_size=5)
        | lists(python_primitives, max_size=5)
        | dictionaries(keys=hashable, values=python_primitives, max_size=3)
    )
)
