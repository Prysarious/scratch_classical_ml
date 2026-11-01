import matplotlib.pyplot as plt
import numpy as np

n_samples = 500

horizontal = np.random.uniform(0, 10, n_samples)
noise = np.random.normal(0, 1.0, n_samples)
vertical = horizontal + noise + 2

w = 0.0
b = 0.0
lr = 0.001
n = len(horizontal)
epochs = 100

epoch_list = []
loss_list = []

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(horizontal, vertical, color='dodgerblue', alpha=0.5, s=20, label='Random Data')
x_line = np.linspace(0, 10, 100)
line, = ax1.plot(x_line, w * x_line + b, color='crimson', lw=2.5, label='Fitted Line')
ax1.set_xlim(0, 10)
ax1.set_ylim(min(vertical) - 2, max(vertical) + 2)
ax1.set_xlabel('Horizontal Features (x)', fontsize=11)
ax1.set_ylabel('Vertical Targets (y)', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left')

loss_line, = ax2.plot([], [], color='purple', lw=2, marker='o', markersize=3, label='MSE Loss')
ax2.set_xlim(1, epochs)
ax2.set_ylim(0, 50)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Mean Squared Error (Loss)', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper right')
ax2.set_title('Loss Function Evolution', fontsize=12, fontweight='bold')

for epoch in range(epochs):
    total_error = 0
    dw = 0
    db = 0
    
    for i in range(n):
        x = horizontal[i]
        y = vertical[i]
        predicted = w * x + b
        error = predicted - y
        
        total_error += (error ** 2) / n
        dw += (2 / n) * error * x
        db += (2 / n) * error
        
    print(f"Epoch {epoch + 1}: Total Error = {total_error:.4f} | w = {w:.4f}, b = {b:.4f}")
    
    epoch_list.append(epoch + 1)
    loss_list.append(total_error)
    
    w = w - lr * dw
    b = b - lr * db
    
    line.set_ydata(w * x_line + b)
    ax1.set_title(f'Epoch {epoch + 1} | w = {w:.3f}, b = {b:.3f}', fontsize=11, fontweight='bold')
    
    loss_line.set_xdata(epoch_list)
    loss_line.set_ydata(loss_list)
    ax2.set_ylim(0, max(loss_list) * 1.1)
    
    plt.draw()
    plt.pause(0.1)

plt.ioff()
print("-" * 50)
print(f"Training Complete! Final Parameters -> w: {w:.4f}, b: {b:.4f}")
plt.show()