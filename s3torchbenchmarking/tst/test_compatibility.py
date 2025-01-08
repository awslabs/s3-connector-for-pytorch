def test_imports():
    from s3torchbenchmarking import datagen
    from s3torchbenchmarking.dataset import benchmark

    assert benchmark is not None
    assert datagen is not None
