#!/usr/bin/env python3
import numpy as np
import sys
sys.path.insert(0, '/workspaces/PyDeepFlow')

from pydeepflow.model import im2col_indices, col2im_indices

# Simple debug case
X = np.ones((1, 3, 3, 1)).astype(np.float32)
print("Original X (all ones):")
print(X[0, :, :, 0])
print(f"Shape: {X.shape}\n")

# im2col
X_col = im2col_indices(X, 2, 2, padding=0, stride=1)
print(f"X_col shape: {X_col.shape}")
print("X_col (each row is a patch):")
print(X_col)
print()

# col2im
X_back = col2im_indices(X_col, X.shape, 2, 2, padding=0, stride=1)
print("Reconstructed X (should show overlap counts):")
print(X_back[0, :, :, 0])
print(f"Shape: {X_back.shape}\n")

# Expected overlap pattern for 3x3 input with 2x2 filter, stride 1:
# - Top-left corner (0,0): covered by 1 patch
# - Top edge (0,1): covered by 2 patches  
# - Left edge (1,0): covered by 2 patches
# - Center (1,1): covered by 4 patches
print("Expected overlap pattern:")
expected = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]])
print(expected)
print()

print("Match:", np.allclose(X_back[0,:,:,0], expected))
