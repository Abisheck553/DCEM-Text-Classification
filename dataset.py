import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import vgg16, resnet50, inception_v3


# 10 classes from the paper (5 proposed + 5 from STD)
CLASSES = [
    "yoga", "teleshopping", "craft", "cooking", "concert",
    "animal", "outlet", "technology", "elearning", "animation",
]

# Extended 15-class version (full STD dataset from paper)
CLASSES_FULL = CLASSES + ["weather", "medical", "sports", "economics", "defenses"]

IMG_SIZE_VGG     = (224, 224)
IMG_SIZE_INCEPTION = (299, 299)

# Each backbone needs its own preprocessing function
PREPROCESS_FN = {
    "vgg16":     vgg16.preprocess_input,
    "resnet50":  resnet50.preprocess_input,
    "inception": inception_v3.preprocess_input,
}


def get_data_generators(
    data_dir: str,
    img_size: tuple = IMG_SIZE_VGG,
    batch_size: int = 32,
    val_split: float = 0.2,
    augment: bool = True,
    backbone: str = "vgg16",
):
    """
    Returns (train_gen, val_gen, class_names) from a directory structured as:
        data_dir/
            class_name/
                img1.jpg
                img2.jpg
    """
    preprocess_fn = PREPROCESS_FN.get(backbone, vgg16.preprocess_input)

    if augment:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            validation_split=val_split,
        )
    else:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            validation_split=val_split,
        )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        validation_split=val_split,
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen, train_gen.class_indices


def get_test_generator(
    data_dir: str,
    img_size: tuple = IMG_SIZE_VGG,
    batch_size: int = 32,
    backbone: str = "vgg16",
):
    """Returns a test generator (no augmentation, no shuffle)."""
    preprocess_fn = PREPROCESS_FN.get(backbone, vgg16.preprocess_input)
    datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return test_gen


def load_single_image(image_path: str, img_size: tuple = IMG_SIZE_VGG, backbone: str = "vgg16") -> np.ndarray:
    """Load and preprocess a single image for prediction."""
    preprocess_fn = PREPROCESS_FN.get(backbone, vgg16.preprocess_input)
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = preprocess_fn(np.expand_dims(arr, axis=0))
    return arr


def create_sample_dataset(base_dir: str, classes: list = None, images_per_class: int = 10):
    """
    Creates a dummy dataset with random noise images for testing.
    Useful when you don't have real data yet.
    """
    if classes is None:
        classes = CLASSES

    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(images_per_class):
            img_path = os.path.join(cls_dir, f"sample_{i}.png")
            if not os.path.exists(img_path):
                dummy = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
                tf.keras.utils.save_img(img_path, dummy)

    print(f"Sample dataset created at: {base_dir}")
    print(f"Classes: {classes}")
    print(f"Images per class: {images_per_class}")
