from tensorflow import keras
from tensorflow.keras import layers


def build_mlp(input_dim=3072, num_classes=10):
    """
    Simple MLP for flattened CIFAR-10 inputs.
    You can modify this to match your assignment architecture.
    """
    model = keras.Sequential([
        layers.Dense(512, activation="relu", input_shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def build_cnn(num_classes=10):
    """
    Simple CNN for CIFAR-10 images (32x32x3).
    """
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation="relu", padding="same", input_shape=(32, 32, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model
