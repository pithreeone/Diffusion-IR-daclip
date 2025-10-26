import re
import matplotlib.pyplot as plt

# Read log files
with open("experiments/universal-ir_251022-092324/val_universal-ir_251022-092325.log", "r") as f:
    log_a = f.read()

path = "experiments/universal-ir_251023-161547/val_universal-ir_251023-161547.log"
with open("experiments/universal-ir_251025-020827/val_universal-ir_251025-020827.log", "r") as f:
    log_b = f.read()

# Extract PSNR values
a = [float(x) for x in re.findall(r"psnr:\s*([0-9.]+)", log_a)]
b = [float(x) for x in re.findall(r"psnr:\s*([0-9.]+)", log_b)]

# Generate independent iteration lists (assuming 20,000 step intervals)
iters_a = [i * 20000 for i in range(1, len(a) + 1)]
iters_b = [i * 20000 for i in range(1, len(b) + 1)]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(iters_a, a, marker='o', label='Model A')
plt.plot(iters_b, b, marker='s', label='Model B')
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.title("PSNR Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("psnr.png")
