def test_imports():
    from s3torchbenchmarking import benchmark, datagen

    assert benchmark is not None
    assert datagen is not None
