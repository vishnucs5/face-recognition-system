import numpy as np
from numba import cuda

# 1. Define the CUDA kernel
@cuda.jit
def vector_add(a, b, c):
    # Calculate unique thread index across the entire grid
    idx = cuda.grid(1)
    
    # Boundary guard
    if idx < a.size:
        c[idx] = a[idx] + b[idx]

# 2. Prepare data on the host (CPU)
N = 100_000
h_a = np.ones(N, dtype=np.float32) * 2.0
h_b = np.ones(N, dtype=np.float32) * 3.0

# 3. Allocate and copy memory to device (GPU)
d_a = cuda.to_device(h_a)
d_b = cuda.to_device(h_b)
d_c = cuda.device_array_like(h_a)

# 4. Configure thread and block dimensions
threads_per_block = 256
blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block

# 5. Launch the kernel: kernel[blocks, threads](arguments)
vector_add[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

# 6. Copy the result back to host (CPU)
h_c = d_c.copy_to_host()

# Verify
print("First 5 results:", h_c[:5])
print("Correct:", np.allclose(h_c, 5.0))