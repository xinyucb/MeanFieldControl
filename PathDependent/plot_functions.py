import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

def plot_evolution_of_density(X1, X2, K, label1, label2, title="Evolution of $A_t$ Density under Two Controls", xlabel=""):
    """
    X1: (M, N+1)
    """
    times = np.arange(np.shape(X1)[1])
    t_normalized = times / times[-1]  # normalize time to [0, 1]
    x_grid = np.linspace(min(X1.min(), X2.min()), max(X1.max(), X2.max()), 200)

    def density_matrix(X):
        dens = np.zeros((len(x_grid), len(times)))
        for i, t in enumerate(times):
            kde = gaussian_kde(X[:, t])
            dens[:, i] = kde(x_grid)
        return dens

    density1 = density_matrix(X1)
    density2 = density_matrix(X2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # α₁ heatmap
    im1 = axes[0].imshow(
        density1, aspect='auto', origin='lower',
        extent=[0, 1, x_grid[0], x_grid[-1]],   # x-axis now 0–1
        cmap='Blues'
    )
    axes[0].axhline(y=K, color='red', linestyle='--', linewidth=2)
    axes[0].set_title(label1)
    axes[0].set_xlabel("Time t")
    
    axes[0].set_ylabel(xlabel)

    # α₂ heatmap
    im2 = axes[1].imshow(
        density2, aspect='auto', origin='lower',
        extent=[0, 1, x_grid[0], x_grid[-1]],   # x-axis now 0–1
        cmap='Blues'
    )
    axes[1].axhline(y=K, color='red', linestyle='--', linewidth=2)
    axes[1].set_title(label2)
    axes[1].set_xlabel("Time t")

    axes[0].annotate(
        "K",
        xy=(0, K),
        xytext=(-5, 0),
        textcoords='offset points',
        color='red',
        fontsize=12,
        va='center',
        ha='right'
    )

    plt.subplots_adjust(right=0.85, wspace=0.1)

    # colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label("Density")

    plt.suptitle(title, y=1.02)
    plt.show()

def plot_one_distribution(X1, K, label1, title="Distribution of " + r"$A_T$"):
    plt.style.use('default')
    plt.figure(figsize=(7, 4.5))

    mean1 = np.mean(X1)
    var1 = np.var(X1)
    print(mean1, var1)
    plt.hist(
        X1, density=True, bins=40, alpha=0.5,
        label=label1 + f" (mean={mean1:.3f}, var={var1:.3f})"
    )

   

    # add vertical line at x = 1.25
    plt.axvline(x=K, color='red', linestyle='--', linewidth=2, label='x = K')

    plt.title(title)
    plt.legend()
    plt.xlabel(r"$A_T$")
    plt.ylabel("Density")

    plt.show()



def plot_two_distribution(X1, X2, K, label1, label2, title="Distribution of " + r"$A_T$"):

    plt.style.use('default')
    plt.figure(figsize=(7, 4.5))

    mean1 = np.mean(X1)
    var1 = np.var(X1)
    print(mean1, var1)
    plt.hist(
        X1, density=True, bins=40, alpha=0.5,
        label=label1 + f" (mean={mean1:.3f}, var={var1:.3f})"
    )

    mean2 = np.mean(X2)
    var2 = np.var(X2)
    print(mean2, var2)
    plt.hist(
        X2, density=True, bins=40, alpha=0.5,
        label=label2 + f" (mean={mean2:.3f}, var={var2:.3f})"
    )

    # add vertical line at x = 1.25
    plt.axvline(x=K, color='red', linestyle='--', linewidth=2, label='x = K')

    plt.title(title)
    plt.legend()
    plt.xlabel(r"$A_T$")
    plt.ylabel("Density")

    plt.show()

def plot_loss(x1, x2, its, N, title="", loss_type="train", label1 =r"$a(t, X_t)$", label2 = r"$a(t, \Delta W)$", save=False):
    """
    
    """
    # # ✅ Path (current path + outputNN）
    base_dir = os.path.dirname(os.path.abspath(__file__))  # current file path
    save_dir = os.path.join(base_dir, "output")

    # check whether path exists and create the path 
    if not os.path.exists(save_dir):
        print(f"The path does not exist. Building the path...{save_dir}")
        os.makedirs(save_dir)
    else:
        print(f"The path exists: {save_dir}")
    save_path = os.path.join(save_dir, f"{title}_{loss_type}_loss.png")

    if loss_type == "validation":
    # df2 = pd.read_csv(f"outputLoss/{read_fileX}_val_loss.csv")
        plt.plot(np.linspace(0, its* N +1, its), list(x1), "-o", alpha=0.5, label=label1)
        plt.plot(np.linspace(0, its * 50 +1, its), list(x2), "-d", alpha=0.5, label=label2)
    else:
        plt.plot(list(x1), alpha=0.5, label=label1)
        plt.plot(list(x2),  alpha=0.5, label=label2)
    
    plt.ylabel(f"{loss_type} loss")
    plt.xlabel("Training Epoch t")
    plt.legend()
    if save:
        plt.savefig(save_path)
    plt.show()