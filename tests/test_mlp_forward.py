"""
Tests for MLP forward pass functionality.
"""

# handle imports
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.nn as nn
from src.models.torch_mlp import MLP


def test_mlp_default_architecture():
    """Test MLP with default architecture."""
    model = MLP(input_size=1024)
    
    # Test forward pass
    x = torch.randn(4, 1024)
    y = model(x)
    
    assert y.shape == (4, 1)
    assert isinstance(y, torch.Tensor)


def test_mlp_custom_hidden_layers():
    """Test MLP with custom hidden layer sizes."""
    hidden_sizes = [256, 128, 64]
    model = MLP(input_size=1024, hidden_layer_sizes=hidden_sizes)
    
    # Test forward pass
    x = torch.randn(4, 1024)
    y = model(x)
    
    assert y.shape == (4, 1)
    assert isinstance(y, torch.Tensor)


def test_mlp_encode_method():
    """Test MLP encode method."""
    model = MLP(input_size=1024, hidden_layer_sizes=[128, 64, 32])
    
    x = torch.randn(4, 1024)
    latent = model.encode(x)
    
    assert latent.shape == (4, 32)  # Last hidden layer size
    assert isinstance(latent, torch.Tensor)


def test_mlp_different_batch_sizes():
    """Test MLP with different batch sizes."""
    model = MLP(input_size=1024)
    
    for batch_size in [1, 8, 32, 128]:
        x = torch.randn(batch_size, 1024)
        y = model(x)
        assert y.shape == (batch_size, 1)


def test_mlp_dropout():
    """Test MLP with dropout."""
    model = MLP(input_size=1024, dropout_prob=0.5)
    model.train()  # Enable training mode for dropout
    
    x = torch.randn(4, 1024)
    y1 = model(x)
    y2 = model(x)
    
    # With dropout, outputs should be different (with high probability)
    assert not torch.allclose(y1, y2, atol=1e-6)


def test_mlp_no_dropout():
    """Test MLP without dropout."""
    model = MLP(input_size=1024, dropout_prob=0.0)
    model.eval()  # Disable training mode
    
    x = torch.randn(4, 1024)
    y1 = model(x)
    y2 = model(x)
    
    # Without dropout, outputs should be identical
    assert torch.allclose(y1, y2)


def test_mlp_single_input():
    """Test MLP with single input."""
    model = MLP(input_size=1024)
    
    x = torch.randn(1, 1024)
    y = model(x)
    
    assert y.shape == (1, 1)


def test_mlp_zero_input():
    """Test MLP with zero input."""
    model = MLP(input_size=1024)
    
    x = torch.zeros(4, 1024)
    y = model(x)
    
    # Should produce some output (due to biases)
    assert y.shape == (4, 1)
    assert not torch.allclose(y, torch.zeros_like(y))


def test_mlp_gradient_flow():
    """Test that gradients flow through the network."""
    model = MLP(input_size=1024)
    x = torch.randn(4, 1024, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    # Check that input gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__])