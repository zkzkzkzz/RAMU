import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 1. GENERATE CLEAN, LINEARLY SEPARABLE DATA
np.random.seed(42)
X_0 = np.random.randn(15, 2) * 0.3 + np.array([1.5, 1.5])  # Class 0 (Red)
X_1 = np.random.randn(15, 2) * 0.3 + np.array([4.0, 4.0])  # Class 1 (Green)
X = np.vstack((X_0, X_1))
y = np.array([0] * 15 + [1] * 15)


# A reusable plotting function to avoid messy duplicated code
def create_mapping_plot(model_instance, title_prefix):
    # Extract the REAL weights and bias learned by the algorithm
    w_learned = model_instance.coef_[0]
    b_learned = model_instance.intercept_[0]

    # Calculate the 1D scores using the learned parameters: z = w1*x1 + w2*x2 + b
    z_0 = np.dot(X_0, w_learned) + b_learned
    z_1 = np.dot(X_1, w_learned) + b_learned
    z_all = np.dot(X, w_learned) + b_learned

    # Generate the math curves based on the learned score range
    z_range = np.linspace(z_all.min() - 2, z_all.max() + 2, 1000)
    sigmoid_curve = 1 / (1 + np.exp(-z_range))
    heaviside_curve = np.where(z_range < 0, 0.0, np.where(z_range > 0, 1.0, 0.5))

    # Setup the 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- PANEL 1: 2D Spatial Input Space ---
    ax1.scatter(X_0[:, 0], X_0[:, 1], color='#e74c3c', s=100, edgecolors='k', label='Class 0 (y=0)', zorder=4)
    ax1.scatter(X_1[:, 0], X_1[:, 1], color='#2ecc71', s=100, edgecolors='k', label='Class 1 (y=1)', zorder=4)

    # Draw the learned decision boundary line (where z = 0)
    x1_boundary = np.array([0, 6])
    x2_boundary = (-w_learned[0] * x1_boundary - b_learned) / w_learned[1]
    ax1.plot(x1_boundary, x2_boundary, color='#34495e', linewidth=2, linestyle='--', label=f'Learned Boundary (z=0)')

    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 6)
    ax1.set_xlabel('Feature $x_1$')
    ax1.set_ylabel('Feature $x_2$')
    ax1.set_title(f"{title_prefix}: 2D Learned Boundary")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')

    # --- PANEL 2: 1D Score Mapping ---
    ax2.plot(z_range, sigmoid_curve, color='#3498db', linewidth=3, label='Sigmoid Curve')
    ax2.step(z_range, heaviside_curve, color='#e74c3c', linewidth=2, linestyle=':', label='Heaviside Step')

    # Map the real points to their 1D locations based on learned scores
    ax2.scatter(z_0, np.zeros_like(z_0), color='#e74c3c', s=100, edgecolors='k', zorder=4)
    ax2.scatter(z_1, np.ones_like(z_1), color='#2ecc71', s=100, edgecolors='k', zorder=4)

    ax2.axvline(0, color='#34495e', linestyle='--', linewidth=1.5)
    ax2.set_xlim(z_range.min(), z_range.max())
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel(f'Learned Score ($z = {w_learned[0]:.2f}x_1 + {w_learned[1]:.2f}x_2 + {b_learned:.2f}$)')
    ax2.set_ylabel('Output Probability')
    ax2.set_title(f"{title_prefix}: Mapped 1D Probability")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left')

    # Draw coupling arrows for the first point of each class
    inv1 = ax1.transData.transform
    inv2 = ax2.transData.transform
    fig_inv = fig.transFigure.inverted().transform

    p0_start = fig_inv(inv1((X_0[0, 0], X_0[0, 1])))
    p0_end = fig_inv(inv2((z_0[0], 0.0)))
    arrow_0 = plt.Annotation("", xy=p0_end, xytext=p0_start, xycoords='figure fraction', textcoords='figure fraction',
                             arrowprops=dict(arrowstyle="->", color="#e74c3c", linestyle="--", alpha=0.6, lw=1.5))
    fig.add_artist(arrow_0)

    p1_start = fig_inv(inv1((X_1[0, 0], X_1[0, 1])))
    p1_end = fig_inv(inv2((z_1[0], 1.0)))
    arrow_1 = plt.Annotation("", xy=p1_end, xytext=p1_start, xycoords='figure fraction', textcoords='figure fraction',
                             arrowprops=dict(arrowstyle="->", color="#2ecc71", linestyle="--", alpha=0.6, lw=1.5))
    fig.add_artist(arrow_1)

    plt.tight_layout()
    print(f"[{title_prefix}] Learned Weights: {w_learned.round(2)} | Learned Bias: {b_learned:.2f}")


# 2. RUN EXPERIMENT 1: CONTROLLED (With Regularization)
# C=1.0 is standard regularization. It stops weights from expanding too much.
controlled_model = LogisticRegression(C=1.0)
controlled_model.fit(X, y)
create_mapping_plot(controlled_model, "Figure 1 (Controlled Sigmoid)")

# 3. RUN EXPERIMENT 2: UNCONTROLLED (No Regularization)
# C=1e9 effectively turns off regularization. max_iter=5000 lets weights blow up.
uncontrolled_model = LogisticRegression(C=1e9, max_iter=5000, solver='saga')
uncontrolled_model.fit(X, y)
create_mapping_plot(uncontrolled_model, "Figure 2 (Uncontrolled Heaviside)")

# Display both windows
plt.show()
