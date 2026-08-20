"""
Cat vs. Dog image classifier - training script (BASELINE).

Trains a small convolutional neural network from scratch on the Kaggle
"cats-and-dogs" dataset (marquis03/cats-and-dogs), which is committed in data/.

No transfer learning and no pretrained weights: every weight in this model is
learned from the 275 training images in this repository.

Usage:
    python train.py

Output:
    model.keras   - the trained model, used later by predict.py
"""

import os
import time

# Silence TensorFlow's C++ INFO/WARNING logs. Must be set before importing
# tensorflow, otherwise the flag is read too late to have any effect.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Every experiment branch changes only the values in this block, so that
# "git diff main" shows exactly what was calibrated.

SEED = 42               # one seed for Python, NumPy and TensorFlow
IMG_SIZE = (128, 128)   # all images are resized to this before training
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
SHUFFLE_BUFFER = 1000   # larger than the dataset, so shuffling is a true full shuffle

TRAIN_DIR = os.path.join("data", "train")
VAL_DIR = os.path.join("data", "val")
CLASS_NAMES = ["cat", "dog"]   # fixed order -> label 0 = cat, label 1 = dog
MODEL_PATH = "model.keras"

EXPERIMENT_NAME = "more-epochs"   # printed in the report, documented in RESULTS.md

LINE = "=" * 64


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seeds() -> None:
    """Make every run of this script produce identical numbers.

    set_random_seed seeds Python's `random`, NumPy and TensorFlow in one call.
    enable_op_determinism additionally forces TensorFlow kernels to use
    deterministic algorithms, so shuffling and weight init cannot drift.
    """
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_dataset(directory: str, shuffle: bool) -> tf.data.Dataset:
    """Build a tf.data pipeline from a folder of class sub-folders.

    The dataset on disk is already split into data/train and data/val, and each
    of them contains one sub-folder per class ("cat", "dog"). That is exactly
    the layout image_dataset_from_directory expects, so no CSV parsing is needed.
    """
    if not os.path.isdir(directory):
        raise SystemExit(
            f"Missing dataset folder: {directory}\n"
            "The dataset is committed in this repository - make sure you cloned "
            "the whole repo and are running train.py from its root."
        )

    dataset = keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",          # labels come from the sub-folder names
        class_names=CLASS_NAMES,    # explicit order, never rely on disk order
        label_mode="binary",        # single 0/1 target per image
        image_size=IMG_SIZE,
        batch_size=None,            # batching is done below, after shuffling
        shuffle=False,
        verbose=False,
    )

    # cache() keeps the decoded images in RAM (the dataset is only ~10 MB), so
    # every JPEG is read and resized exactly once instead of once per epoch.
    dataset = dataset.cache()

    # Shuffle single images *before* batching, so every epoch sees different
    # batch compositions. Doing it after batching would only reorder fixed
    # batches, which trains noticeably worse.
    if shuffle:
        dataset = dataset.shuffle(
            SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
        )

    # prefetch() overlaps data preparation with training - pure speed, no effect
    # on the maths.
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def count_labels(dataset: tf.data.Dataset) -> dict[str, int]:
    """Count how many images of each class a dataset contains."""
    labels = np.concatenate([y.numpy().ravel() for _, y in dataset])
    return {name: int((labels == i).sum()) for i, name in enumerate(CLASS_NAMES)}


def majority_class_baseline(counts: dict[str, int]) -> tuple[str, float]:
    """Accuracy of the dumbest possible model: always predict the biggest class.

    Our data is imbalanced (far more dogs than cats), so a model that learned
    nothing at all can still look decent. Any real model must clearly beat this.
    """
    majority_class = max(counts, key=counts.get)
    accuracy = counts[majority_class] / sum(counts.values())
    return majority_class, accuracy


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model() -> keras.Model:
    """A small sequential CNN built from basic Keras layers.

    Three convolution blocks progressively halve the resolution while doubling
    the number of feature maps (128 -> 64 -> 32 -> 16 pixels, 16 -> 32 -> 64
    filters), then a dense head turns those features into one probability.

    This is the deliberately plain BASELINE: no augmentation and no dropout, so
    the experiment branches have something honest to improve on.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=IMG_SIZE + (3,)),
            # Pixels arrive as 0-255 integers; neural nets train far better on
            # small floats, so scale them into the 0-1 range.
            layers.Rescaling(1.0 / 255),

            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            # One output neuron + sigmoid = probability that the image is a dog.
            layers.Dense(1, activation="sigmoid"),
        ],
        name="cats_vs_dogs_cnn",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",   # the standard loss for two-class problems
        metrics=["accuracy"],
    )
    return model


# ----------------------------------------------------------------------------
# Readable console output
# ----------------------------------------------------------------------------
class EpochReporter(keras.callbacks.Callback):
    """Prints one aligned line per epoch instead of Keras' progress bars."""

    def on_epoch_end(self, epoch, logs=None):
        print(
            f"  epoch {epoch + 1:>2}/{EPOCHS}"
            f"   loss {logs['loss']:.4f}  acc {logs['accuracy']:.4f}"
            f"   |   val_loss {logs['val_loss']:.4f}  val_acc {logs['val_accuracy']:.4f}"
        )


def print_header(train_counts: dict[str, int], val_counts: dict[str, int]) -> None:
    print(f"\n{LINE}\n  CAT vs DOG  -  training ({EXPERIMENT_NAME})\n{LINE}")
    print(f"  images     train {sum(train_counts.values()):>3}   "
          f"(cat {train_counts['cat']}, dog {train_counts['dog']})")
    print(f"             val   {sum(val_counts.values()):>3}   "
          f"(cat {val_counts['cat']}, dog {val_counts['dog']})")
    print(f"  input      {IMG_SIZE[0]}x{IMG_SIZE[1]} RGB   batch {BATCH_SIZE}   "
          f"epochs {EPOCHS}   seed {SEED}")


def print_summary(history, baseline_class: str, baseline_acc: float,
                  elapsed: float) -> None:
    val_acc = history.history["val_accuracy"]
    best_epoch = int(np.argmax(val_acc)) + 1
    best_acc = float(val_acc[best_epoch - 1])
    train_acc = float(history.history["accuracy"][best_epoch - 1])

    print(f"\n{LINE}\n  RESULTS ({EXPERIMENT_NAME})\n{LINE}")
    print(f"  majority-class baseline      {baseline_acc:6.2%}"
          f"   (always answer \"{baseline_class}\")")
    print(f"  best val_accuracy            {best_acc:6.2%}   (epoch {best_epoch})")
    print(f"  train_accuracy at that epoch {train_acc:6.2%}")
    # Reported in percentage POINTS: the honest way to compare two accuracies.
    print(f"  lift over baseline           {(best_acc - baseline_acc) * 100:+6.2f} pts")
    print(f"{LINE}")
    print(f"  trained in {elapsed:.1f}s on CPU   |   saved best epoch -> {MODEL_PATH}")
    print(f"  next step: python predict.py\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    set_seeds()

    train_ds = load_dataset(TRAIN_DIR, shuffle=True)
    val_ds = load_dataset(VAL_DIR, shuffle=False)

    train_counts = count_labels(train_ds)
    val_counts = count_labels(val_ds)
    baseline_class, baseline_acc = majority_class_baseline(val_counts)

    print_header(train_counts, val_counts)

    model = build_model()
    print(f"  model      {model.count_params():,} trainable parameters, "
          f"trained from scratch\n{LINE}")

    # Keep the weights from the epoch with the best validation accuracy, not the
    # last epoch, which may already be overfitting.
    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_accuracy", mode="max", save_best_only=True
    )

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, EpochReporter()],
        shuffle=False,   # the tf.data pipeline already reshuffles every epoch
        verbose=0,       # our EpochReporter prints the progress instead
    )
    elapsed = time.time() - start

    print_summary(history, baseline_class, baseline_acc, elapsed)


if __name__ == "__main__":
    main()
