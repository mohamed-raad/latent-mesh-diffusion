import torch, time

# Test GPU speed
t0 = time.time()
x = torch.randn(1000, 128, device="cuda")
y = torch.randn(1000, 128, device="cuda")
for _ in range(100):
    z = (x @ y.T).softmax(dim=-1)
torch.cuda.synchronize()
print(f"GPU matmul + softmax x100: {time.time()-t0:.3f}s")

# Check VRAM
print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
