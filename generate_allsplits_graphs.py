import matplotlib.pyplot as plt
import numpy as np


# The PCE scores averaged across the 4 dataset splits for each model
all_splits_results = {
    'MobileNetV2\n(Authors)':  0.813,
    'MobileNetV2\n(Scratch)':  0.641,  
    'MobileNetV3-Large':       0.650,
    'EfficientNet-B0':         0.754,
}

# Contrasting colours for bar chart
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

models = list(all_splits_results.keys())
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, list(all_splits_results.values()), color=colors, width=0.5, edgecolor='black', linewidth=0.5)

for bar, score in zip(bars, all_splits_results.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
ax.set_ylabel('PCE (Probability of Correct Event)', fontsize=13)
ax.set_title('PCE Comparison — Average Across All 4 Splits', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.95)

#Authors baseline height for comparison
ax.axhline(y=0.813, color='#2196F3', linestyle='--', alpha=0.5, label='Authors avg all splits (0.813)') 

ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graph_pce_allsplits.png', dpi=150)
print('Saved graph_pce_allsplits.png')
plt.show()