"""Basic tests for quant-fairness."""

import pytest


def test_import():
    """Verify package imports."""
    from quant_fairness import __version__, quick_sweep, simulate_int4
    assert __version__ == "0.1.0"


def test_get_num_layers():
    """Test layer detection."""
    from quant_fairness.sweep import get_num_layers

    class MockConfig:
        n_layer = 12

    class MockModel:
        config = MockConfig()

        def state_dict(self):
            return {}

    model = MockModel()
    assert get_num_layers(model) == 12


def test_int4_simulation():
    """Test INT4 quantization simulation."""
    import torch
    from quant_fairness.quantize import simulate_int4

    class MockModel:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.randn(10, 10))

        def named_parameters(self):
            yield 'h.0.mlp.weight', self.weight

    model = MockModel()
    original = model.weight.clone()

    simulate_int4(model, protect_layers={0})

    # Layer 0 protected - should be unchanged
    assert torch.allclose(model.weight, original)

    # Now without protection
    model.weight.data = original.clone()
    simulate_int4(model, protect_layers=set())

    # Should be quantized (different from original)
    assert not torch.allclose(model.weight, original)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
