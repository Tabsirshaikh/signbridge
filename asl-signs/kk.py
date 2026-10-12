# debug_processed.py
import numpy as np

X = np.load("processed/X_train.npy", mmap_mode='r')
print(f"Shape     : {X.shape}")
print(f"Max value : {X.max():.4f}")
print(f"Min value : {X.min():.4f}")
print(f"Mean      : {X.mean():.4f}")
print(f"Zeros %   : {(X == 0).mean() * 100:.1f}%")
print(f"\nSample sequence [0, 0, :10]:\n{X[0, 0, :10]}")