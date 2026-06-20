import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf

from dataset import load_single_image, CLASSES


def predict_image(
    model_path: str,
    image_path: str,
    class_names: list = None,
    output_dir: str = "output",
    top_k: int = 3,
) -> dict:
    """
    Run DCEM prediction on a single image.
    Returns the top-k predicted classes with confidence scores.
    """
    if class_names is None:
        class_names = CLASSES

    model = tf.keras.models.load_model(model_path)

    _, h, w, _ = model.input_shape
    img_array = load_single_image(image_path, img_size=(h, w))

    probs = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]

    results = [
        {"class": class_names[i] if i < len(class_names) else f"class_{i}",
         "confidence": float(probs[i]) * 100}
        for i in top_indices
    ]

    print(f"\n[DCEM Prediction] Image: {os.path.basename(image_path)}")
    print(f"{'Rank':<6} {'Class':<20} {'Confidence':>12}")
    print("-" * 40)
    for rank, r in enumerate(results, 1):
        print(f"{rank:<6} {r['class']:<20} {r['confidence']:>11.2f}%")

    # Visualise prediction
    _save_prediction_plot(image_path, results, probs, class_names, output_dir)

    return {"image": image_path, "predictions": results, "all_probs": probs.tolist()}


def predict_batch(
    model_path: str,
    image_dir: str,
    class_names: list = None,
    output_dir: str = "output",
):
    """Run predictions on all images inside a directory."""
    if class_names is None:
        class_names = CLASSES

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not image_paths:
        print(f"No images found in: {image_dir}")
        return []

    print(f"\n[DCEM] Processing {len(image_paths)} images...")
    all_results = []
    for path in sorted(image_paths):
        result = predict_image(model_path, path, class_names, output_dir, top_k=1)
        all_results.append(result)

    return all_results


def _save_prediction_plot(image_path, results, all_probs, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: image
    axes[0].imshow(img)
    axes[0].axis("off")
    top_cls = results[0]["class"]
    top_conf = results[0]["confidence"]
    axes[0].set_title(f"Prediction: {top_cls}\nConfidence: {top_conf:.1f}%", fontsize=12)

    # Right: bar chart of all class probabilities
    short_names = [n[:12] for n in class_names]
    colors = ["#2196F3" if i != np.argmax(all_probs) else "#F44336" for i in range(len(class_names))]
    axes[1].barh(short_names, all_probs * 100, color=colors)
    axes[1].set_xlabel("Confidence (%)")
    axes[1].set_xlim([0, 110])
    axes[1].set_title("Class Probabilities")
    axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle("DCEM — Text Image Classification", fontsize=13)
    plt.tight_layout()

    fname = os.path.splitext(os.path.basename(image_path))[0] + "_prediction.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Prediction plot saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCEM Prediction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved .h5 model")
    parser.add_argument("--image",      type=str, default=None,  help="Single image path")
    parser.add_argument("--image_dir",  type=str, default=None,  help="Directory of images (batch mode)")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--top_k",      type=int, default=3)
    args = parser.parse_args()

    if args.image:
        predict_image(args.model_path, args.image, output_dir=args.output_dir, top_k=args.top_k)
    elif args.image_dir:
        predict_batch(args.model_path, args.image_dir, output_dir=args.output_dir)
    else:
        print("Provide --image <path> or --image_dir <dir>")
        sys.exit(1)
