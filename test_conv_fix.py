#!/usr/bin/env python3
import numpy as np
import sys
sys.path.insert(0, '/workspaces/PyDeepFlow')

from pydeepflow.model import im2col_indices, col2im_indices, ConvLayer

def test_im2col_basic():
    """Test basic im2col functionality"""
    print("=" * 60)
    print("Test 1: Basic im2col shape and values")
    print("=" * 60)
    
    # Simple test case: 2x2 image, 2x2 filter, 1 channel
    X = np.arange(1, 5).reshape(1, 2, 2, 1).astype(np.float32)
    print(f"Input X shape: {X.shape}")
    print(f"Input X:\n{X[0, :, :, 0]}")
    
    X_col = im2col_indices(X, 2, 2, padding=0, stride=1)
    print(f"\nX_col shape: {X_col.shape}")
    print(f"Expected shape: (1, 4) -> (1 output position, 2*2*1 patch size)")
    print(f"X_col:\n{X_col}")
    
    # Verify shape
    assert X_col.shape == (1, 4), f"Shape mismatch: {X_col.shape} != (1, 4)"
    
    # Verify values - should be the flattened image
    expected = np.array([[1, 2, 3, 4]])
    assert np.allclose(X_col, expected), f"Values mismatch:\n{X_col}\n!=\n{expected}"
    
    print("✓ Test 1 PASSED\n")

def test_im2col_stride():
    """Test im2col with stride"""
    print("=" * 60)
    print("Test 2: im2col with stride=2")
    print("=" * 60)
    
    # 4x4 image, 2x2 filter, stride=2
    X = np.arange(1, 17).reshape(1, 4, 4, 1).astype(np.float32)
    print(f"Input X shape: {X.shape}")
    print(f"Input X:\n{X[0, :, :, 0]}")
    
    X_col = im2col_indices(X, 2, 2, padding=0, stride=2)
    print(f"\nX_col shape: {X_col.shape}")
    print(f"Expected shape: (4, 4) -> (2*2 output positions, 2*2*1 patch size)")
    print(f"X_col:\n{X_col}")
    
    # Verify shape
    assert X_col.shape == (4, 4), f"Shape mismatch: {X_col.shape} != (4, 4)"
    
    # Verify first patch (top-left)
    expected_first = np.array([1, 2, 5, 6])
    assert np.allclose(X_col[0], expected_first), f"First patch mismatch:\n{X_col[0]}\n!=\n{expected_first}"
    
    print("✓ Test 2 PASSED\n")

def test_col2im_roundtrip():
    """Test col2im reverses im2col (with gradient accumulation)"""
    print("=" * 60)
    print("Test 3: col2im roundtrip")
    print("=" * 60)
    
    # Use simpler test with ones to verify overlap pattern
    X = np.ones((1, 3, 3, 1)).astype(np.float32)
    print(f"Input X shape: {X.shape} (all ones)")
    
    # Forward with 2x2 filter, stride=1
    X_col = im2col_indices(X, 2, 2, padding=0, stride=1)
    print(f"X_col shape: {X_col.shape}")
    
    # Backward (gradient should accumulate due to overlapping patches)
    X_reconstructed = col2im_indices(X_col, X.shape, 2, 2, padding=0, stride=1)
    print(f"Reconstructed X shape: {X_reconstructed.shape}")
    
    # Expected overlap pattern for 3x3 input with 2x2 filter:
    # Each position shows how many times it's visited by the sliding window
    expected = np.array([[1., 2., 1.],
                         [2., 4., 2.],
                         [1., 2., 1.]], dtype=np.float32).reshape(1, 3, 3, 1)
    
    print(f"\nExpected overlap pattern:\n{expected[0,:,:,0]}")
    print(f"Reconstructed:\n{X_reconstructed[0,:,:,0]}")
    
    assert np.allclose(X_reconstructed, expected), "Roundtrip failed!"
    
    print("✓ Test 3 PASSED\n")

def test_convlayer_forward():
    """Test ConvLayer forward pass"""
    print("=" * 60)
    print("Test 4: ConvLayer forward pass")
    print("=" * 60)
    
    # Create a simple conv layer
    conv = ConvLayer(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
    
    # Input: batch of 2, 8x8 image, 3 channels
    X = np.random.randn(2, 8, 8, 3).astype(np.float32)
    print(f"Input shape: {X.shape}")
    
    # Forward pass
    out = conv.forward(X)
    print(f"Output shape: {out.shape}")
    print(f"Expected shape: (2, 8, 8, 8)")
    
    # Verify shape
    assert out.shape == (2, 8, 8, 8), f"Shape mismatch: {out.shape} != (2, 8, 8, 8)"
    
    print("✓ Test 4 PASSED\n")

def test_convlayer_backward():
    """Test ConvLayer backward pass"""
    print("=" * 60)
    print("Test 5: ConvLayer backward pass")
    print("=" * 60)
    
    conv = ConvLayer(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
    
    X = np.random.randn(2, 8, 8, 3).astype(np.float32)
    print(f"Input shape: {X.shape}")
    
    # Forward
    out = conv.forward(X)
    print(f"Forward output shape: {out.shape}")
    
    # Backward with gradient
    dOut = np.random.randn(*out.shape).astype(np.float32)
    dX = conv.backward(dOut)
    
    print(f"Gradient dX shape: {dX.shape}")
    print(f"Expected shape: {X.shape}")
    
    # Verify shapes
    assert dX.shape == X.shape, f"Gradient shape mismatch: {dX.shape} != {X.shape}"
    assert conv.grads['dW'].shape == conv.params['W'].shape, "Weight gradient shape mismatch"
    assert conv.grads['db'].shape == (8,), "Bias gradient shape mismatch"
    
    print(f"Weight gradient shape: {conv.grads['dW'].shape}")
    print(f"Bias gradient shape: {conv.grads['db'].shape}")
    
    print("✓ Test 5 PASSED\n")

def test_gradient_numerical():
    """Numerical gradient check for ConvLayer"""
    print("=" * 60)
    print("Test 6: Numerical gradient check")
    print("=" * 60)
    
    # Small layer for faster computation
    conv = ConvLayer(in_channels=2, out_channels=3, kernel_size=2, stride=1, padding=0)
    
    X = np.random.randn(1, 3, 3, 2).astype(np.float32) * 0.1
    
    # Forward
    out = conv.forward(X)
    
    # Backward with simple gradient (sum of outputs)
    dOut = np.ones_like(out)
    dX = conv.backward(dOut)
    
    # Numerical gradient for input
    eps = 1e-5
    numerical_grad = np.zeros_like(X)
    
    print("Computing numerical gradient (this may take a moment)...")
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            for k in range(X.shape[3]):
                X_plus = X.copy()
                X_plus[0, i, j, k] += eps
                out_plus = conv.forward(X_plus).sum()
                
                X_minus = X.copy()
                X_minus[0, i, j, k] -= eps
                out_minus = conv.forward(X_minus).sum()
                
                numerical_grad[0, i, j, k] = (out_plus - out_minus) / (2 * eps)
    
    # Restore the forward cache
    _ = conv.forward(X)
    analytical_grad = conv.backward(dOut)
    
    # Compare
    rel_error = np.max(np.abs(numerical_grad - analytical_grad) / (np.abs(numerical_grad) + np.abs(analytical_grad) + 1e-8))
    print(f"Max relative error: {rel_error}")
    print(f"Threshold: 1e-3")
    
    assert rel_error < 1e-3, f"Gradient check failed! Error: {rel_error}"
    
    print("✓ Test 6 PASSED\n")

if __name__ == "__main__":
    try:
        test_im2col_basic()
        test_im2col_stride()
        test_col2im_roundtrip()
        test_convlayer_forward()
        test_convlayer_backward()
        test_gradient_numerical()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe im2col/col2im fix is working correctly!")
        print("ConvLayer forward and backward passes are functional.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
