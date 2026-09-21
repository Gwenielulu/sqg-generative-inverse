from sqg_inverse.config import merge


def test_merge():
    merged = merge({"model": {"width": 128, "depth": 3}}, {"model": {"depth": 4}})
    assert merged == {"model": {"width": 128, "depth": 4}}
