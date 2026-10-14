"""Quaternion rotation and fusion checks for the four public entry points."""

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = (
    "BCI_14_7513_sota",
    "BCI_2a9_7922_sota",
    "BCI_IVA5_8479_sota",
    "BCI_IVNO7_8293_sota",
)


@pytest.fixture(params=ENTRY_POINTS)
def ops(request):
    path = ROOT / request.param / "core_qnn" / "quaternion_ops.py"
    spec = importlib.util.spec_from_file_location("ops_" + request.param, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def product(left, right):
    """Hamilton product for individual quaternions on the last axis."""
    a, b, c, d = left.unbind(-1)
    e, f, g, h = right.unbind(-1)
    return torch.stack((
        a * e - b * f - c * g - d * h,
        a * f + b * e + c * h - d * g,
        a * g - b * h + c * e + d * f,
        a * h + b * g - c * f + d * e,
    ), dim=-1)


def rotate(ops, x, weights, quaternion_format=True, scale=None, bias=None):
    components = [value.reshape(1, 1) for value in weights.unbind()]
    zero = torch.zeros_like(components[0])
    return ops.quaternion_linear_rotation(
        x, zero, *components, bias=bias,
        quaternion_format=quaternion_format, scale=scale,
    )


def test_identity_preserves_all_four_components(ops):
    x = torch.tensor([[2., 1., 3., 4.], [-3., 0., 0., 0.]], dtype=torch.double)
    weights = torch.tensor([1., 0., 0., 0.], dtype=torch.double)
    torch.testing.assert_close(rotate(ops, x, weights), x, rtol=0, atol=0)


def test_scalar_survives_arbitrary_rotation(ops):
    x = torch.tensor([[2., 0., 0., 0.], [-3., 0., 0., 0.]], dtype=torch.double)
    weights = torch.tensor([0.3, -0.7, 0.2, 0.4], dtype=torch.double)
    torch.testing.assert_close(rotate(ops, x, weights), x, rtol=0, atol=0)


@pytest.mark.parametrize("batch_shape", [(7,), (2, 3)])
def test_rotation_matches_hamilton_sandwich(ops, batch_shape):
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(*batch_shape, 4, generator=generator, dtype=torch.double)
    weights = torch.tensor([0.3, -0.7, 0.2, 0.4], dtype=torch.double)
    unit = weights / weights.norm()
    conjugate = unit * unit.new_tensor([1., -1., -1., -1.])
    expected = product(product(unit, x), conjugate)
    torch.testing.assert_close(rotate(ops, x, weights), expected, rtol=1e-12, atol=1e-12)


def test_positive_z_rotation_direction(ops):
    x = torch.tensor([[2., 1., 0., 0.]], dtype=torch.double)
    weights = torch.tensor([math.sqrt(0.5), 0., 0., math.sqrt(0.5)], dtype=torch.double)
    expected = torch.tensor([[2., 0., 1., 0.]], dtype=torch.double)
    torch.testing.assert_close(rotate(ops, x, weights), expected, rtol=1e-12, atol=1e-12)


def test_rotation_preserves_norm(ops):
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(13, 4, generator=generator, dtype=torch.double)
    weights = torch.tensor([0.003, -0.007, 0.002, 0.004], dtype=torch.double)
    actual = rotate(ops, x, weights)
    torch.testing.assert_close(actual.norm(dim=-1), x.norm(dim=-1), rtol=1e-12, atol=1e-12)


def test_zero_weights_are_identity_with_finite_gradients(ops):
    x = torch.tensor([[2., 1., 3., 4.]], dtype=torch.double, requires_grad=True)
    weights = torch.zeros(4, dtype=torch.double, requires_grad=True)
    actual = rotate(ops, x, weights)
    torch.testing.assert_close(actual, x, rtol=0, atol=0)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(weights.grad).all()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_rotation_gradcheck(ops):
    x = torch.tensor([[2., 1., 3., 4.]], dtype=torch.double, requires_grad=True)
    weights = torch.tensor([0.3, -0.7, 0.2, 0.4], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: rotate(ops, a, b), (x, weights))


@pytest.mark.parametrize("quaternion_format", [False, True])
def test_optional_scale_applies_to_vector_part(ops, quaternion_format):
    x = torch.tensor([[2., 1., 3., 4.]], dtype=torch.double)
    weights = torch.tensor([0.3, -0.7, 0.2, 0.4], dtype=torch.double)
    unit = weights / weights.norm()
    conjugate = unit * unit.new_tensor([1., -1., -1., -1.])
    expected = product(product(unit, x), conjugate)
    expected[..., 1:] *= 2.5
    if not quaternion_format:
        x = x[..., 1:]
        expected = expected[..., 1:]
    bias = torch.full_like(x[0], 0.25)
    scale = torch.tensor([[2.5]], dtype=torch.double)
    actual = rotate(ops, x, weights, quaternion_format, scale=scale, bias=bias)
    torch.testing.assert_close(actual, expected + bias, rtol=1e-12, atol=1e-12)


def load_entry_point(name):
    local_roots = {"modules", "dataloader", "datasets_", "Adver_network", "utils",
                   "Weight", "model", "core_qnn", "centroid_align", "logmap"}
    for loaded_name in list(sys.modules):
        if loaded_name.split(".", 1)[0] in local_roots:
            del sys.modules[loaded_name]
    entry_dir = ROOT / name
    old_path = sys.path[:]
    sys.path.insert(0, str(entry_dir))
    try:
        spec = importlib.util.spec_from_file_location("test_entry_" + name, entry_dir / "main.py")
        module = importlib.util.module_from_spec(spec)
        module.args = argparse.Namespace(window_size=5)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


@pytest.mark.parametrize("entry_name", ENTRY_POINTS)
def test_temporal_projection_receives_gradient(entry_name):
    module = load_entry_point(entry_name)
    torch.manual_seed(17)
    head = module.QuaternionFusionHead(d_model=16, dropout=0).eval()
    channel = torch.randn(2, 3, 16)
    temporal = torch.randn(2, 3, 16)
    head(channel, temporal).square().sum().backward()
    gradients = [parameter.grad for parameter in head.temporal_proj.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
