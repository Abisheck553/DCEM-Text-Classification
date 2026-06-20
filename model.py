import tensorflow as tf
from tensorflow.keras import layers, models, applications


def _freeze_base(base_model):
    for layer in base_model.layers:
        layer.trainable = False


def build_dcem(num_classes: int, backbone: str = "vgg16", input_shape: tuple = (224, 224, 3)) -> models.Model:
    """
    Deep Convolutional Encoder Model (DCEM).
    Encoder -> Bottleneck (latent representation) -> Classifier (softmax).
    Backbone choices: 'vgg16', 'resnet50', 'inception' (InceptionV3).
    """
    if backbone == "inception":
        input_shape = (299, 299, 3)

    inputs = layers.Input(shape=input_shape, name="input_image")

    if backbone == "vgg16":
        base = applications.VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    elif backbone == "resnet50":
        base = applications.ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif backbone == "inception":
        base = applications.InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose 'vgg16', 'resnet50', or 'inception'.")

    _freeze_base(base)

    # Encoder
    encoded = base(inputs, training=False)

    # Bottleneck — latent representation
    x = layers.GlobalAveragePooling2D(name="bottleneck_pool")(encoded)
    x = layers.Dense(512, activation="relu", name="bottleneck_dense")(x)
    x = layers.BatchNormalization(name="bottleneck_bn")(x)
    x = layers.Dropout(0.5, name="bottleneck_drop")(x)

    # 4-module convolutional classifier head (dense equivalent, per paper spec)
    for i, units in enumerate([256, 128, 64], start=1):
        x = layers.Dense(units, activation="relu", name=f"cls_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"cls_bn_{i}")(x)
        if i < 3:
            x = layers.Dropout(0.3, name=f"cls_drop_{i}")(x)

    # Softmax encoder-based classifier
    outputs = layers.Dense(num_classes, activation="softmax", name="classification")(x)

    return models.Model(inputs, outputs, name=f"DCEM_{backbone}")


def build_multitask_dcem(num_classes: int, input_shape: tuple = (224, 224, 3)) -> models.Model:
    """
    Multi-task DCEM with two output heads:
      - 'classification': action-text class (softmax, num_classes)
      - 'detection':      text presence binary flag (sigmoid)
    Uses VGG16 shared encoder as described in the paper.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    base = applications.VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    _freeze_base(base)
    shared = base(inputs, training=False)

    # Shared bottleneck
    gap = layers.GlobalAveragePooling2D(name="shared_gap")(shared)
    shared_dense = layers.Dense(512, activation="relu", name="shared_dense")(gap)
    shared_bn = layers.BatchNormalization(name="shared_bn")(shared_dense)
    shared_drop = layers.Dropout(0.5, name="shared_drop")(shared_bn)

    # Classification branch
    cls = layers.Dense(128, activation="relu", name="cls_dense_1")(shared_drop)
    cls = layers.BatchNormalization(name="cls_bn_1")(cls)
    cls = layers.Dense(64, activation="relu", name="cls_dense_2")(cls)
    cls_out = layers.Dense(num_classes, activation="softmax", name="classification")(cls)

    # Text detection branch (binary: text present / absent)
    det = layers.Dense(128, activation="relu", name="det_dense_1")(shared_drop)
    det = layers.BatchNormalization(name="det_bn_1")(det)
    det_out = layers.Dense(1, activation="sigmoid", name="detection")(det)

    return models.Model(inputs, [cls_out, det_out], name="DCEM_Multitask")


def unfreeze_top_layers(model: models.Model, num_layers: int = 4):
    """Fine-tune the last `num_layers` of the backbone encoder."""
    # The backbone is the 2nd layer (index 1) in DCEM models
    backbone = model.layers[1]
    for layer in backbone.layers[-num_layers:]:
        layer.trainable = True
