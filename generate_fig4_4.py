# save as: generate_fig4_4.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

cm = np.array([[325, 24],
               [68, 277]])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted\nAuthentic', 'Predicted\nTampered'],
            yticklabels=['Actual\nAuthentic', 'Actual\nTampered'],
            annot_kws={"size": 20}, ax=ax)

ax.set_title('Figure 4.4: Confusion Matrix for MobileNetV2 (Threshold: 0.70)', fontsize=14)
ax.set_xlabel('Predicted Class', fontsize=12)
ax.set_ylabel('Actual Class', fontsize=12)

# Add percentage labels
total_auth = 325 + 24
total_tamp = 68 + 277
ax.text(0.5, 0.35, f'TN: {325/total_auth*100:.1f}%', ha='center', va='center', fontsize=11, color='white', fontweight='bold', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('figure_4_4.png', dpi=300)
plt.show()
print("Saved as figure_4_4.png")