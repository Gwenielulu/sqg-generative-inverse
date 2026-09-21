from sqg_inverse.inverse.conditioning import needs_grad


def test_gradient_methods():
    assert not needs_grad("vanilla")
    assert not needs_grad("projection")
    assert needs_grad("dps")
    assert needs_grad("dps_plus")
    assert needs_grad("mcg")
