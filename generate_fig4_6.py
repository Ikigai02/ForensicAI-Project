# save as: generate_fig4_6.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Style settings
box_style = dict(boxstyle='round,pad=0.4', facecolor='#D6EAF8', edgecolor='#2C3E50', linewidth=2)
green_box = dict(boxstyle='round,pad=0.4', facecolor='#D5F5E3', edgecolor='#27AE60', linewidth=2)
yellow_box = dict(boxstyle='round,pad=0.4', facecolor='#FEF9E7', edgecolor='#F39C12', linewidth=2)
red_box = dict(boxstyle='round,pad=0.4', facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=2)
process_box = dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB', edgecolor='#2980B9', linewidth=2)
diamond_box = dict(boxstyle='round,pad=0.3', facecolor='#F5EEF8', edgecolor='#8E44AD', linewidth=2)

# Step 1: Input
ax.text(6, 9.2, 'Receipt Image Uploaded', ha='center', va='center', fontsize=12, fontweight='bold', bbox=box_style)

# Arrow
ax.annotate('', xy=(6, 8.5), xytext=(6, 8.85), arrowprops=dict(arrowstyle='->', lw=2))

# Step 2: ELA
ax.text(6, 8.2, 'ELA Processing (Quality 75)', ha='center', va='center', fontsize=11, bbox=process_box)

# Arrow
ax.annotate('', xy=(6, 7.5), xytext=(6, 7.85), arrowprops=dict(arrowstyle='->', lw=2))

# Step 3: Resize + Preprocess
ax.text(6, 7.2, 'Resize 128×128 + preprocess_input [-1, 1]', ha='center', va='center', fontsize=11, bbox=process_box)

# Arrow
ax.annotate('', xy=(6, 6.5), xytext=(6, 6.85), arrowprops=dict(arrowstyle='->', lw=2))

# Step 4: MobileNetV2
ax.text(6, 6.2, 'MobileNetV2 Prediction → Score (0 to 1)', ha='center', va='center', fontsize=11, fontweight='bold', bbox=box_style)

# Arrow
ax.annotate('', xy=(6, 5.5), xytext=(6, 5.85), arrowprops=dict(arrowstyle='->', lw=2))

# Decision diamond
ax.text(6, 5.2, 'CNN Confidence Score', ha='center', va='center', fontsize=11, fontweight='bold', bbox=diamond_box)

# Three branches
# Left: Authentic
ax.annotate('', xy=(2.5, 3.8), xytext=(4.5, 4.85), arrowprops=dict(arrowstyle='->', lw=2))
ax.text(3.2, 4.5, 'Score < 0.40', ha='center', va='center', fontsize=10, color='#27AE60', fontweight='bold')
ax.text(2.5, 3.3, 'AUTHENTIC\n(REAL)', ha='center', va='center', fontsize=12, fontweight='bold', bbox=green_box)
ax.text(2.5, 2.5, 'No evidence of\ntampering detected', ha='center', va='center', fontsize=9, color='gray')

# Middle: Suspicious
ax.annotate('', xy=(6, 3.8), xytext=(6, 4.85), arrowprops=dict(arrowstyle='->', lw=2))
ax.text(6, 4.5, '0.40 ≤ Score ≤ 0.70', ha='center', va='center', fontsize=10, color='#F39C12', fontweight='bold')
ax.text(6, 3.3, 'SUSPICIOUS\n(MANUAL REVIEW)', ha='center', va='center', fontsize=12, fontweight='bold', bbox=yellow_box)
ax.text(6, 2.5, 'Borderline case\nflagged for expert review', ha='center', va='center', fontsize=9, color='gray')

# Right: Tampered
ax.annotate('', xy=(9.5, 3.8), xytext=(7.5, 4.85), arrowprops=dict(arrowstyle='->', lw=2))
ax.text(8.8, 4.5, 'Score > 0.70', ha='center', va='center', fontsize=10, color='#E74C3C', fontweight='bold')
ax.text(9.5, 3.3, 'TAMPERED\n(FAKE)', ha='center', va='center', fontsize=12, fontweight='bold', bbox=red_box)
ax.text(9.5, 2.5, 'Strong forensic indicators\nof manipulation', ha='center', va='center', fontsize=9, color='gray')

ax.set_title('Figure 4.6: Three-Tier Verdict Classification System Decision Flow', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figure_4_6.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved as figure_4_6.png")