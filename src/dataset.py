import numpy as np
from pathlib import Path


def load_assignment2_data(root_dir: str | None = None):
    """
    Load CIFAR-10 data from the Assignment2Data folder, as provided in the assignment.

    Expected structure (relative to project root by default):

        Assignment2Data/
            X_train.npy
            y_train.npy
            X_test.npy
            y_test.npy

    Args:
        root_dir: optional path to project root. If None, use current working directory.

    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    if root_dir is None:
        root = Path(".")
    else:
        root = Path(root_dir)

    data_dir = root / "Assignment2Data"

    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    return (X_train, y_train), (X_test, y_test)
