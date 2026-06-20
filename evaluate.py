import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf

from dataset import get_test_generator, CLASSES


def evaluate_model(
    model_path: str,
    test_dir: str,
    class_names: list = None,
    output_dir: str = "output",
    batch_size: int = 32,
):
    if class_names is None:
        class_names = CLASSES

    os.makedirs(output_dir, exist_ok=True)

    model = tf.keras.models.load_model(model_path)

    # Determine image size from model input shape
    _, h, w, _ = model.input_shape
    img_size = (h, w)

    backbone = "inception" if max(h, w) == 299 else ("resnet50" if "resnet" in model_path else "vgg16")
    test_gen = get_test_generator(test_dir, img_size=img_size, batch_size=batch_size, backbone=backbone)

    print("[Evaluate] Running predictions...")
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes
    labels = list(test_gen.class_indices.keys())

    # ── Classification report ─────────────────────────────────────────────────
    print("\n── Classification Report ──────────────────────────────")
    report = classification_report(y_true, y_pred, target_names=labels, digits=2)
    print(report)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # ── Per-class Precision / Recall / F1 (matches Table 2 in paper) ─────────
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(labels)))
    print(f"\n{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 55)
    for i, name in enumerate(labels):
        print(f"{name:<20} {precision[i]*100:>9.1f}% {recall[i]*100:>9.1f}% {f1[i]*100:>9.1f}%")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    _plot_confusion_matrix(y_true, y_pred, labels, output_dir)

    # ── ROC curve (one-vs-rest, Figure 3 from paper) ─────────────────────────
    _plot_roc_curve(y_true, y_prob, labels, output_dir)

    # ── Training history (if passed via history object) ───────────────────────
    overall_acc = np.mean(y_pred == y_true) * 100
    print(f"\n[Evaluate] Overall Accuracy: {overall_acc:.2f}%")

    return {
        "accuracy": overall_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_names": labels,
    }


def _plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix — DCEM Model")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Confusion matrix saved: {path}")


def _plot_roc_curve(y_true, y_prob, labels, output_dir):
    """ROC curve (one-vs-rest) — reproduces Fig 3 from the paper."""
    n_classes = len(labels)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    mean_tpr = np.zeros(100)
    mean_fpr = np.linspace(0, 1, 100)

    for i, (cls_name, color) in enumerate(zip(labels, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{cls_name} (AUC={roc_auc:.2f})")
        mean_tpr += np.interp(mean_fpr, fpr, tpr)

    # Macro-average ROC
    mean_tpr /= n_classes
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, "k--", lw=2.5, label=f"Macro avg (AUC={mean_auc:.2f})")
    ax.plot([0, 1], [0, 1], "r:", lw=1)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve — DCEM Model")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] ROC curve saved: {path}")


def plot_training_history(history_phase1, history_phase2=None, output_dir: str = "output"):
    """Plots accuracy and loss curves for both training phases."""
    os.makedirs(output_dir, exist_ok=True)

    acc = history_phase1.history["accuracy"]
    val_acc = history_phase1.history["val_accuracy"]
    loss = history_phase1.history["loss"]
    val_loss = history_phase1.history["val_loss"]

    if history_phase2:
        acc += history_phase2.history["accuracy"]
        val_acc += history_phase2.history["val_accuracy"]
        loss += history_phase2.history["loss"]
        val_loss += history_phase2.history["val_loss"]
        phase2_start = len(history_phase1.history["accuracy"])
    else:
        phase2_start = None

    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, val_metric, title in [
        (axes[0], acc, val_acc, "Model Accuracy"),
        (axes[1], loss, val_loss, "Model Loss"),
    ]:
        ax.plot(epochs_range, metric, label="Train")
        ax.plot(epochs_range, val_metric, label="Validation")
        if phase2_start:
            ax.axvline(x=phase2_start, color="gray", linestyle="--", label="Fine-tune start")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("DCEM Training History", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Training history saved: {path}")


def plot_bar_comparison(results: dict, output_dir: str = "output"):
    """
    Bar chart comparing Precision / Recall / F1 per class.
    Reproduces Fig 2 (Comparative analysis) from the paper.
    """
    os.makedirs(output_dir, exist_ok=True)
    names = results["class_names"]
    precision = results["precision"] * 100
    recall = results["recall"] * 100
    f1 = results["f1"] * 100

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label="Precision", color="#2196F3")
    ax.bar(x,         recall,    width, label="Recall",    color="#4CAF50")
    ax.bar(x + width, f1,        width, label="F1-Score",  color="#FF9800")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim([0, 110])
    ax.set_title("Comparison with Various Approaches — DCEM Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "bar_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Bar comparison saved: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DCEM model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_dir",   type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model_path,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    plot_bar_comparison(results, output_dir=args.output_dir)
