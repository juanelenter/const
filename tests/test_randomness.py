import numpy as np
import pytest
import torch

from src.utils.utils import RNGContext, set_seed


def test_RNGContext():
    """Test that the RNGContext context manager sets and resets the RNG states correctly."""
    seed = 42
    with RNGContext(seed):
        # Generate some random numbers using numpy and torch
        np_array_1 = np.random.rand(3, 3)
        torch_tensor_1 = torch.rand(3, 3)

    with RNGContext(seed):
        # Generate some more random numbers using numpy and torch
        np_array_2 = np.random.rand(3, 3)
        torch_tensor_2 = torch.rand(3, 3)

    # Check that the generated numbers are the same within the context
    assert np.allclose(np_array_1, np_array_2)
    assert torch.all(torch.eq(torch_tensor_1, torch_tensor_2))

    # Generate some more random numbers using numpy and torch after the context
    np_array_3 = np.random.rand(3, 3)
    torch_tensor_3 = torch.rand(3, 3)

    # Check that the generated numbers are different after the context
    assert not np.allclose(np_array_2, np_array_3)
    assert not torch.all(torch.eq(torch_tensor_2, torch_tensor_3))


def test_set_seed():
    """Test that set_seed sets the seed for the random number generators correctly."""
    seed = 42
    set_seed(seed)

    # Generate some random numbers using numpy and torch
    np_array_1 = np.random.rand(3, 3)
    torch_tensor_1 = torch.rand(3, 3)

    set_seed(seed)

    # Generate some more random numbers using numpy and torch
    np_array_2 = np.random.rand(3, 3)
    torch_tensor_2 = torch.rand(3, 3)

    # Check that the generated numbers are the same after setting the seed twice
    assert np.allclose(np_array_1, np_array_2)
    assert torch.all(torch.eq(torch_tensor_1, torch_tensor_2))
