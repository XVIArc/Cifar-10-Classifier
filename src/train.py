import argparse
import numpy as np
from tensorflow import keras

from dataset import load_assignment2_data
from models import build_mlp, build_cnn


def main(args):
    (X_train, y_train), (X_test, y_test) = load_assignment2_data()

    # 根据模型类型决定是否 flatten
    if args.model == "mlp":
        # (N, 32, 32, 3) -> (N, 3072)
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        model = build_mlp(input_dim=X_train_flat.shape[1], num_classes=10)
        x_tr, x_te = X_train_flat, X_test_flat
    else:
        # CNN 直接用原始形状
        model = build_cnn(num_classes=10)
        x_tr, x_te = X_train, X_test

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_tr, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.1,
        verbose=2
    )

    loss, acc = model.evaluate(x_te, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on CIFAR-10 (Assignment2 data)")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    main(args)
