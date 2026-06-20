"""
Provisioning A Pre-Trained Network Model for Extracting Text from Images
========================================================================
Implementation of the Deep Convolutional Encoder Model (DCEM) as described in:
  "Monitoring and Analysis of Students' Live Behaviour Using Machine Learning"
  V Abisheck, G Abinesh, S Abiram — IEEE ICERCS'23

Usage examples
--------------
# 1. Create a sample dataset (for testing without real data)
python main.py --mode demo

# 2. Train with VGG16 backbone
python main.py --mode train --data_dir data/train --backbone vgg16

# 3. Train with ResNet50
python main.py --mode train --data_dir data/train --backbone resnet50

# 4. Evaluate on test set
python main.py --mode evaluate --model_path output/dcem_vgg16_final.h5 --test_dir data/test

# 5. Predict a single image
python main.py --mode predict --model_path output/dcem_vgg16_final.h5 --image path/to/image.jpg
"""

import os
import argparse

from dataset import create_sample_dataset, CLASSES
from train import train, train_multitask
from evaluate import evaluate_model, plot_training_history, plot_bar_comparison
from predict import predict_image, predict_batch


def demo_mode(output_dir: str = "output"):
    """End-to-end demo with synthetic data — no real dataset required."""
    print("=" * 60)
    print("  DCEM Text Classification — Demo Mode")
    print("=" * 60)

    # Step 1: Create dummy dataset
    data_dir = "data/sample"
    test_dir = "data/sample_test"
    print("\n[1/4] Creating sample dataset...")
    create_sample_dataset(data_dir, classes=CLASSES, images_per_class=20)
    create_sample_dataset(test_dir, classes=CLASSES, images_per_class=5)

    # Step 2: Train
    print("\n[2/4] Training DCEM (VGG16 backbone, 3 epochs demo)...")
    model, h1, h2 = train(
        data_dir=data_dir,
        backbone="vgg16",
        num_classes=len(CLASSES),
        epochs=3,
        fine_tune_epochs=2,
        batch_size=4,
        output_dir=output_dir,
    )

    # Step 3: Plot history
    print("\n[3/4] Plotting training history...")
    plot_training_history(h1, h2, output_dir=output_dir)

    # Step 4: Evaluate
    print("\n[4/4] Evaluating on test set...")
    model_path = os.path.join(output_dir, "dcem_vgg16_final.h5")
    results = evaluate_model(
        model_path=model_path,
        test_dir=test_dir,
        class_names=CLASSES,
        output_dir=output_dir,
        batch_size=4,
    )
    plot_bar_comparison(results, output_dir=output_dir)

    print("\n" + "=" * 60)
    print("  Demo complete! Check the 'output/' folder for plots.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="DCEM — Pre-Trained Network Model for Text Extraction from Images"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate", "predict", "predict_batch", "demo"],
        help="Operation mode",
    )
    parser.add_argument("--data_dir",    type=str,  default=None, help="Training data directory")
    parser.add_argument("--test_dir",    type=str,  default=None, help="Test data directory")
    parser.add_argument("--model_path",  type=str,  default=None, help="Path to saved model (.h5)")
    parser.add_argument("--image",       type=str,  default=None, help="Single image for prediction")
    parser.add_argument("--image_dir",   type=str,  default=None, help="Directory of images (batch predict)")
    parser.add_argument("--backbone",    type=str,  default="vgg16", choices=["vgg16", "resnet50", "inception"])
    parser.add_argument("--num_classes", type=int,  default=10)
    parser.add_argument("--epochs",      type=int,  default=50)
    parser.add_argument("--fine_tune_epochs", type=int, default=20)
    parser.add_argument("--batch_size",  type=int,  default=32)
    parser.add_argument("--output_dir",  type=str,  default="output")
    parser.add_argument("--multitask",   action="store_true", help="Use multi-task DCEM")
    parser.add_argument("--top_k",       type=int,  default=3)
    args = parser.parse_args()

    if args.mode == "demo":
        demo_mode(args.output_dir)

    elif args.mode == "train":
        if not args.data_dir:
            raise ValueError("--data_dir is required for train mode")
        if args.multitask:
            train_multitask(
                data_dir=args.data_dir,
                num_classes=args.num_classes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
            )
        else:
            model, h1, h2 = train(
                data_dir=args.data_dir,
                backbone=args.backbone,
                num_classes=args.num_classes,
                epochs=args.epochs,
                fine_tune_epochs=args.fine_tune_epochs,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
            )
            plot_training_history(h1, h2, output_dir=args.output_dir)

    elif args.mode == "evaluate":
        if not args.model_path or not args.test_dir:
            raise ValueError("--model_path and --test_dir are required for evaluate mode")
        results = evaluate_model(
            model_path=args.model_path,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
        plot_bar_comparison(results, output_dir=args.output_dir)

    elif args.mode == "predict":
        if not args.model_path or not args.image:
            raise ValueError("--model_path and --image are required for predict mode")
        predict_image(args.model_path, args.image, output_dir=args.output_dir, top_k=args.top_k)

    elif args.mode == "predict_batch":
        if not args.model_path or not args.image_dir:
            raise ValueError("--model_path and --image_dir are required for predict_batch mode")
        predict_batch(args.model_path, args.image_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
