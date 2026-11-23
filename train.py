import os
import argparse
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

from model import build_dcem, build_multitask_dcem, unfreeze_top_layers
from dataset import get_data_generators, CLASSES, IMG_SIZE_VGG, IMG_SIZE_INCEPTION


def get_callbacks(checkpoint_path: str, log_dir: str, best_so_far: float = None) -> list:
    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            initial_value_threshold=best_so_far,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    try:
        import tensorboard  # noqa: F401
        cb_list.append(callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
    except ImportError:
        pass
    return cb_list


def train(
    data_dir: str,
    backbone: str = "vgg16",
    num_classes: int = 10,
    epochs: int = 50,
    fine_tune_epochs: int = 20,
    batch_size: int = 32,
    output_dir: str = "output",
):
    os.makedirs(output_dir, exist_ok=True)

    img_size = IMG_SIZE_INCEPTION if backbone == "inception" else IMG_SIZE_VGG

    print(f"\n[DCEM] Backbone: {backbone.upper()}")
    print(f"[DCEM] Image size: {img_size}")
    print(f"[DCEM] Classes: {num_classes}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_gen, val_gen, class_indices = get_data_generators(
        data_dir, img_size=img_size, batch_size=batch_size, backbone=backbone
    )
    print(f"[DCEM] Class mapping: {class_indices}")

    # ── Phase 1: Train classifier head (encoder frozen) ───────────────────────
    model = build_dcem(num_classes=num_classes, backbone=backbone, input_shape=(*img_size, 3))
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = os.path.join(output_dir, f"dcem_{backbone}_best.h5")
    log_dir = os.path.join(output_dir, "logs", backbone)

    print("\n[Phase 1] Training classifier head with frozen encoder...")
    history_phase1 = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=get_callbacks(checkpoint_path, log_dir),
    )

    # ── Phase 2: Fine-tune top encoder layers ────────────────────────────────
    print("\n[Phase 2] Fine-tuning top layers of encoder...")
    unfreeze_top_layers(model, num_layers=4)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),  # lower LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    best_phase1 = max(history_phase1.history.get("val_accuracy", [0.0]))
    history_phase2 = model.fit(
        train_gen,
        epochs=fine_tune_epochs,
        validation_data=val_gen,
        callbacks=get_callbacks(checkpoint_path, log_dir, best_so_far=best_phase1),
    )

    # Save final model
    final_path = os.path.join(output_dir, f"dcem_{backbone}_final.h5")
    model.save(final_path)
    print(f"\n[DCEM] Model saved to: {final_path}")

    return model, history_phase1, history_phase2


def train_multitask(
    data_dir: str,
    num_classes: int = 10,
    epochs: int = 50,
    batch_size: int = 32,
    output_dir: str = "output",
):
    """
    Train the multi-task DCEM (classification + text detection).

    EXPERIMENTAL: the dataset has no text-presence annotations, so every image is
    given detection label 1. The detection head therefore learns nothing useful
    until real labels are added. The classification head trains normally.
    """
    print("[DCEM] Warning: multitask detection labels are placeholders (all 1).")
    os.makedirs(output_dir, exist_ok=True)

    train_gen, val_gen, class_indices = get_data_generators(
        data_dir, img_size=IMG_SIZE_VGG, batch_size=batch_size
    )

    model = build_multitask_dcem(num_classes=num_classes)
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss={
            "classification": "categorical_crossentropy",
            "detection": "binary_crossentropy",
        },
        loss_weights={"classification": 1.0, "detection": 0.5},
        metrics={"classification": "accuracy", "detection": "accuracy"},
    )

    # For multitask, wrap generator to also produce detection labels
    def multitask_gen(gen):
        for x, y_cls in gen:
            # Binary detection label: 1 if any text class, simplified as all=1 here
            y_det = tf.ones((tf.shape(x)[0], 1), dtype=tf.float32)
            yield x, {"classification": y_cls, "detection": y_det}

    checkpoint_path = os.path.join(output_dir, "dcem_multitask_best.h5")
    log_dir = os.path.join(output_dir, "logs", "multitask")

    history = model.fit(
        multitask_gen(train_gen),
        steps_per_epoch=len(train_gen),
        epochs=epochs,
        validation_data=multitask_gen(val_gen),
        validation_steps=len(val_gen),
        callbacks=get_callbacks(checkpoint_path, log_dir),
    )

    model.save(os.path.join(output_dir, "dcem_multitask_final.h5"))
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DCEM Text Classification Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--backbone", type=str, default="vgg16", choices=["vgg16", "resnet50", "inception"])
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fine_tune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--multitask", action="store_true", help="Use multitask DCEM model")
    args = parser.parse_args()

    if args.multitask:
        train_multitask(
            data_dir=args.data_dir,
            num_classes=args.num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
    else:
        train(
            data_dir=args.data_dir,
            backbone=args.backbone,
            num_classes=args.num_classes,
            epochs=args.epochs,
            fine_tune_epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
