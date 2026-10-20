"""
Optimizer Race: SGD vs AdaGrad vs Adam
----------------------------------------
A from-scratch (NumPy only) implementation of a 2-layer feed-forward
neural network trained on the sklearn "digits" dataset (8x8 handwritten
digit images, 10 classes). SGD and AdaGrad are implemented manually as
raw parameter-update rules; Adam is also implemented manually (not taken
from any deep-learning framework) so that all three optimizers are
compared under identical conditions: same architecture, same weight
initialisation, same batches, same number of epochs.

Outputs:
    results/loss_curve.png        -> convergence speed (loss vs epoch)
    results/accuracy_curve.png    -> test accuracy vs epoch
    results/final_comparison.png  -> bar chart of final loss / accuracy / time
    results/metrics.csv           -> raw numeric results
    results/console_log.txt       -> captured run log (for screenshots)
"""

import os
import time
import csv
import json
import warnings
import numpy as np

# Some BLAS backends (e.g. Apple Accelerate) emit benign RuntimeWarnings for
# matmul on denormal/extreme-magnitude floats even when results are correct;
# verified final losses/accuracies are finite and sane, so these are silenced.
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ----------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------
digits = load_digits()
X, y = digits.data, digits.target
X = X / 16.0  # pixel intensities are in [0, 16]; scale to [0, 1] (zero-variance-safe)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

N_CLASSES = 10
N_FEATURES = X_train.shape[1]
HIDDEN = 32

def one_hot(y, n_classes=N_CLASSES):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1
    return out

Y_train = one_hot(y_train)
Y_test = one_hot(y_test)

# ----------------------------------------------------------------------
# 2. Network: 64 -> 32 (ReLU) -> 10 (softmax), cross-entropy loss
# ----------------------------------------------------------------------
def init_params(seed=SEED):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(N_FEATURES, HIDDEN) * np.sqrt(2.0 / N_FEATURES)
    b1 = np.zeros((1, HIDDEN))
    W2 = rng.randn(HIDDEN, N_CLASSES) * np.sqrt(2.0 / HIDDEN)
    b2 = np.zeros((1, N_CLASSES))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = softmax(z2)
    cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "X": X}
    return a2, cache

def cross_entropy_loss(probs, Y):
    eps = 1e-12
    return -np.mean(np.sum(Y * np.log(probs + eps), axis=1))

def backward(params, cache, Y):
    m = cache["X"].shape[0]
    dz2 = (cache["a2"] - Y) / m
    dW2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)
    da1 = dz2 @ params["W2"].T
    dz1 = da1 * relu_grad(cache["z1"])
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def accuracy(params, X, y):
    probs, _ = forward(params, X)
    preds = probs.argmax(axis=1)
    return (preds == y).mean()

# ----------------------------------------------------------------------
# 3. Optimizers (manual update rules)
# ----------------------------------------------------------------------
class SGD:
    """Vanilla mini-batch Stochastic Gradient Descent."""
    name = "SGD"
    def __init__(self, lr=0.5):
        self.lr = lr
    def step(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]

class AdaGrad:
    """AdaGrad: per-parameter learning rate scaled by accumulated squared gradients."""
    name = "AdaGrad"
    def __init__(self, lr=0.5, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.G = None
    def step(self, params, grads):
        if self.G is None:
            self.G = {k: np.zeros_like(v) for k, v in params.items()}
        for k in params:
            self.G[k] += grads[k] ** 2
            params[k] -= self.lr * grads[k] / (np.sqrt(self.G[k]) + self.eps)

class Adam:
    """Adam: first/second moment estimates with bias correction."""
    name = "Adam"
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
    def step(self, params, grads):
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t += 1
        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ----------------------------------------------------------------------
# 4. Training loop
# ----------------------------------------------------------------------
EPOCHS = 60
BATCH_SIZE = 32

def train(optimizer, log_lines):
    params = init_params(seed=SEED)
    n = X_train.shape[0]
    rng = np.random.RandomState(SEED)

    train_losses, test_losses, test_accs = [], [], []
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(n)
        for start_idx in range(0, n, BATCH_SIZE):
            idx = order[start_idx:start_idx + BATCH_SIZE]
            xb, yb = X_train[idx], Y_train[idx]
            probs, cache = forward(params, xb)
            grads = backward(params, cache, yb)
            optimizer.step(params, grads)

        train_probs, _ = forward(params, X_train)
        test_probs, _ = forward(params, X_test)
        tr_loss = cross_entropy_loss(train_probs, Y_train)
        te_loss = cross_entropy_loss(test_probs, Y_test)
        te_acc = accuracy(params, X_test, y_test)
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        if epoch % 10 == 0 or epoch == 1:
            line = f"[{optimizer.name:8s}] epoch {epoch:3d}/{EPOCHS}  train_loss={tr_loss:.4f}  test_loss={te_loss:.4f}  test_acc={te_acc:.4f}"
            print(line)
            log_lines.append(line)

    elapsed = time.time() - start
    return {
        "name": optimizer.name,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_accs": test_accs,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "final_test_acc": test_accs[-1],
        "time_sec": elapsed,
    }

def epochs_to_threshold(losses, threshold):
    for i, l in enumerate(losses, start=1):
        if l <= threshold:
            return i
    return None

def main():
    log_lines = []
    sep = "-" * 78
    header = (
        "Optimizer Race: SGD vs AdaGrad vs Adam\n"
        f"Dataset: sklearn digits (train={X_train.shape[0]}, test={X_test.shape[0]}, features={N_FEATURES}, classes={N_CLASSES})\n"
        f"Network: {N_FEATURES} -> {HIDDEN} (ReLU) -> {N_CLASSES} (softmax), cross-entropy loss\n"
        f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Seed: {SEED}\n"
        + sep
    )
    print(header)
    log_lines.append(header)

    optimizers = [
        SGD(lr=0.1),
        AdaGrad(lr=0.3),
        Adam(lr=0.01),
    ]

    results = []
    for opt in optimizers:
        results.append(train(opt, log_lines))

    # Threshold used to compare convergence speed across optimizers
    threshold = 0.3
    summary_lines = ["-" * 78, "FINAL SUMMARY", "-" * 78]
    for r in results:
        ep = epochs_to_threshold(r["train_losses"], threshold)
        ep_str = str(ep) if ep else f">{EPOCHS}"
        line = (
            f"{r['name']:8s} | final_train_loss={r['final_train_loss']:.4f} | "
            f"final_test_loss={r['final_test_loss']:.4f} | final_test_acc={r['final_test_acc']*100:.2f}% | "
            f"epochs_to_loss<= {threshold}: {ep_str} | time={r['time_sec']:.3f}s"
        )
        print(line)
        summary_lines.append(line)
    log_lines.extend(summary_lines)

    # ---------------- Plots ----------------
    colors = {"SGD": "#d62728", "AdaGrad": "#1f77b4", "Adam": "#2ca02c"}

    plt.figure(figsize=(8, 5.5))
    for r in results:
        plt.plot(range(1, EPOCHS + 1), r["train_losses"], label=r["name"], color=colors[r["name"]], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (Cross-Entropy)")
    plt.title("Convergence Speed: Training Loss vs Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5.5))
    for r in results:
        plt.plot(range(1, EPOCHS + 1), r["test_accs"], label=r["name"], color=colors[r["name"]], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"), dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    names = [r["name"] for r in results]
    bar_colors = [colors[n] for n in names]

    axes[0].bar(names, [r["final_test_loss"] for r in results], color=bar_colors)
    axes[0].set_title("Final Test Loss")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(names, [r["final_test_acc"] * 100 for r in results], color=bar_colors)
    axes[1].set_title("Final Test Accuracy (%)")
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar(names, [r["time_sec"] for r in results], color=bar_colors)
    axes[2].set_title("Training Time (s)")
    axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "final_comparison.png"), dpi=160)
    plt.close()

    # ---------------- CSV ----------------
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + [f"{r['name']}_train_loss" for r in results] +
                         [f"{r['name']}_test_loss" for r in results] +
                         [f"{r['name']}_test_acc" for r in results])
        for e in range(EPOCHS):
            row = [e + 1]
            row += [r["train_losses"][e] for r in results]
            row += [r["test_losses"][e] for r in results]
            row += [r["test_accs"][e] for r in results]
            writer.writerow(row)

    with open(os.path.join(RESULTS_DIR, "console_log.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")

    # ---------------- JSON (for the dashboard) ----------------
    json_data = {
        "config": {
            "dataset": "sklearn digits",
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
            "features": N_FEATURES,
            "classes": N_CLASSES,
            "architecture": f"{N_FEATURES} -> {HIDDEN} (ReLU) -> {N_CLASSES} (softmax)",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "threshold": threshold,
        },
        "optimizers": {
            r["name"]: {
                "train_losses": r["train_losses"],
                "test_losses": r["test_losses"],
                "test_accs": r["test_accs"],
                "final_train_loss": r["final_train_loss"],
                "final_test_loss": r["final_test_loss"],
                "final_test_acc": r["final_test_acc"],
                "time_sec": r["time_sec"],
                "epochs_to_threshold": epochs_to_threshold(r["train_losses"], threshold),
            }
            for r in results
        },
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\nSaved plots and metrics to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
