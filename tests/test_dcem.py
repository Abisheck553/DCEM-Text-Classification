"""Fast checks that run without the dataset, trained weights, or internet access."""

import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import CLASSES, create_sample_dataset, get_test_generator  # noqa: E402
from model import build_dcem, unfreeze_top_layers  # noqa: E402
from predict import _infer_backbone, predict_image  # noqa: E402


def test_classes_match_keras_label_order(tmp_path):
    # flow_from_directory assigns indices alphabetically; CLASSES must agree.
    create_sample_dataset(str(tmp_path), classes=list(reversed(CLASSES)), images_per_class=1)
    gen = get_test_generator(str(tmp_path), batch_size=2)
    assert list(gen.class_indices) == CLASSES
    assert len(set(CLASSES)) == 10


@pytest.mark.parametrize("backbone", ["vgg16", "resnet50"])
def test_dcem_output_shape(backbone):
    model = build_dcem(num_classes=10, backbone=backbone, weights=None)
    assert model.output_shape == (None, 10)
    assert model.name == f"DCEM_{backbone}"


def test_encoder_starts_frozen_and_top_layers_unfreeze():
    model = build_dcem(num_classes=10, backbone="vgg16", weights=None)
    encoder = model.layers[1]
    assert not any(layer.trainable for layer in encoder.layers)
    unfreeze_top_layers(model, num_layers=4)
    assert all(layer.trainable for layer in encoder.layers[-4:])
    assert not any(layer.trainable for layer in encoder.layers[:-4])


def test_infer_backbone():
    vgg = build_dcem(num_classes=10, backbone="vgg16", weights=None)
    res = build_dcem(num_classes=10, backbone="resnet50", weights=None)
    assert _infer_backbone(vgg, "anything.h5") == "vgg16"
    assert _infer_backbone(res, "model.h5") == "resnet50"


def test_predict_returns_correct_label_for_argmax(tmp_path):
    # A model whose output always favours index 2 must be reported as CLASSES[2].
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    logits = tf.keras.layers.Dense(10, kernel_initializer="zeros",
                                   bias_initializer=tf.keras.initializers.Constant(
                                       [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]))(x)
    model = tf.keras.Model(inputs, tf.keras.layers.Softmax()(logits), name="DCEM_vgg16")
    model_path = tmp_path / "toy.h5"
    model.save(model_path)

    image_path = tmp_path / "img.png"
    tf.keras.utils.save_img(str(image_path), (np.random.rand(64, 64, 3) * 255).astype("uint8"))

    result = predict_image(str(model_path), str(image_path), output_dir=str(tmp_path), top_k=1)
    assert result["predictions"][0]["class"] == CLASSES[2] == "concert"
