"""
Cat vs. Dog image classifier - prediction script.

Loads the model produced by train.py and classifies images the model has never
seen during training (the "foreign input" required by the assignment).

Usage:
    python predict.py                        # classify every image in test-images/
    python predict.py photo.jpg other.png    # classify the images you pass in
"""

import os
import sys

# Must be set before importing tensorflow, or the flag is read too late.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow import keras

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Note there is no image size here: it is read from the trained model itself (see
# model_image_size), so changing IMG_SIZE in train.py can never leave predict.py
# resizing images to the wrong dimensions.
CLASS_NAMES = ["cat", "dog"] # label 0 = cat, label 1 = dog - must match train.py
MODEL_PATH = "model.keras"

TEST_DIR = "test-images"
VALID_SUFFIXES = (".jpg", ".jpeg", ".png")

LINE = "=" * 64


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_trained_model() -> keras.Model:
    """Load model.keras, with a helpful message if it does not exist yet."""
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"No trained model found at '{MODEL_PATH}'.\n"
            "The model is not committed to git on purpose - it is generated.\n"
            "Run this first:   python train.py"
        )
    return keras.models.load_model(MODEL_PATH)


def collect_image_paths(args: list[str]) -> list[str]:
    """Decide which images to classify: the given paths, or all of test-images/."""
    if args:
        missing = [path for path in args if not os.path.isfile(path)]
        if missing:
            raise SystemExit("File(s) not found: " + ", ".join(missing))
        return args

    if not os.path.isdir(TEST_DIR):
        raise SystemExit(f"Folder '{TEST_DIR}/' is missing - it should be part of the repo.")

    # sorted() keeps the demo output in a stable, predictable order
    paths = sorted(
        os.path.join(TEST_DIR, name)
        for name in os.listdir(TEST_DIR)
        if name.lower().endswith(VALID_SUFFIXES)
    )
    if not paths:
        raise SystemExit(f"No images ({', '.join(VALID_SUFFIXES)}) found in '{TEST_DIR}/'.")
    return paths


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def model_image_size(model: keras.Model) -> tuple[int, int]:
    """Ask the model which image size it was trained on.

    The model's input shape is (batch, height, width, channels), so reading it
    keeps predict.py correct automatically - no constant here to forget to update
    when train.py changes.
    """
    _, height, width, _ = model.input_shape
    return height, width


def load_image(path: str, img_size: tuple[int, int]) -> np.ndarray:
    """Read one image from disk into the exact shape the model expects.

    Note there is no division by 255 here: the Rescaling layer is *inside* the
    model, so scaling happens automatically and cannot get out of sync with
    training - a very common source of silently wrong predictions.
    """
    image = keras.utils.load_img(path, target_size=img_size)  # also handles resizing
    return keras.utils.img_to_array(image)


def predict_all(model: keras.Model, paths: list[str]) -> list[float]:
    """Return P(dog) for every image, using a single batched forward pass."""
    img_size = model_image_size(model)
    batch = np.stack([load_image(path, img_size) for path in paths])
    probabilities = model.predict(batch, verbose=0)
    return [float(p) for p in probabilities.ravel()]


def interpret(prob_dog: float) -> tuple[str, float]:
    """Turn one sigmoid output into a human answer plus its confidence.

    The model has a single output neuron = P(dog). If it is below 0.5 the model
    is effectively voting for "cat", and its confidence is the mirrored value.
    """
    if prob_dog >= 0.5:
        return "dog", prob_dog
    return "cat", 1.0 - prob_dog


def confidence_bar(prob_dog: float, width: int = 20) -> str:
    """A tiny ASCII gauge: left end = certain cat, right end = certain dog."""
    filled = int(round(prob_dog * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def print_summary_table(paths: list[str], probabilities: list[float]) -> None:
    """Compact recap of every prediction: image | prediction | confidence."""
    names = [os.path.basename(path) for path in paths]
    width = max(len(name) for name in names)

    print(f"\n{LINE}\n  SUMMARY\n{LINE}")
    print(f"  {'image'.ljust(width)}   {'prediction':<11} confidence")
    print(f"  {'-' * width}   {'-' * 11} {'-' * 10}")
    for name, prob_dog in zip(names, probabilities):
        label, confidence = interpret(prob_dog)
        print(f"  {name.ljust(width)}   {label:<11} {confidence:>7.1%}")
    print(f"{LINE}\n")


def main() -> None:
    paths = collect_image_paths(sys.argv[1:])
    model = load_trained_model()

    print(f"\n{LINE}\n  CAT vs DOG  -  prediction on unseen images\n{LINE}")
    height, width = model_image_size(model)
    print(f"  model      {MODEL_PATH} ({model.count_params():,} parameters, "
          f"{height}x{width} input)")
    print(f"  images     {len(paths)} file(s)"
          f"{'' if sys.argv[1:] else f' from {TEST_DIR}/'}\n{LINE}")

    probabilities = predict_all(model, paths)

    for path, prob_dog in zip(paths, probabilities):
        label, confidence = interpret(prob_dog)
        print(f"  {os.path.basename(path)}")
        print(f"      {confidence_bar(prob_dog)}  cat <-> dog")
        print(f"      -> {label.upper()}   confidence {confidence:.1%}"
              f"   (P(dog) = {prob_dog:.3f})")

    # The table is only useful when several images are classified at once.
    if len(paths) > 1:
        print_summary_table(paths, probabilities)
    else:
        print()


if __name__ == "__main__":
    main()
