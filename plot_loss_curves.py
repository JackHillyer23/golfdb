import pandas as pd
import matplotlib.pyplot as plt

v2 = pd.read_csv('mobilenetv2_loss.csv')
v3 = pd.read_csv('mobilenetv3_loss.csv')
eff = pd.read_csv('efficientnet_loss.csv')

window = 50  # smoothing window
fig, ax = plt.subplots(figsize=(12, 6))

# Plot raw loss faintly in background
ax.plot(v2['iteration'], v2['loss'], alpha=0.35, color='#2196F3')
ax.plot(v3['iteration'], v3['loss'], alpha=0.35, color='#FF9800')
ax.plot(eff['iteration'], eff['loss'], alpha=0.35, color='#4CAF50')


ax.plot(v2['iteration'], v2['loss'].rolling(window=window, min_periods=1).mean(), label='MobileNetV2', color='#2196F3', linewidth=2)
ax.plot(v3['iteration'], v3['loss'].rolling(window=window, min_periods=1).mean(), label='MobileNetV3-Large', color='#FF9800', linewidth=2)
ax.plot(eff['iteration'], eff['loss'].rolling(window=window, min_periods=1).mean(), label='EfficientNet-B0', color='#4CAF50', linewidth=2)
ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('Training Loss', fontsize=13)
ax.set_title('Training Loss Curves Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loss_comparison.png', dpi=150)
plt.show()