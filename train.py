"""
Cat vs. Dog image classifier - training script (COMBINED / synthesis).

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
EPOCHS = 30   # augmentation makes every epoch harder, so the model needs more of them
LEARNING_RATE = 1e-3
SHUFFLE_BUFFER = 1000   # larger than the dataset, so shuffling is a true full shuffle

TRAIN_DIR = os.path.join("data", "train")
VAL_DIR = os.path.join("data", "val")
CLASS_NAMES = ["cat", "dog"]   # fixed order -> label 0 = cat, label 1 = dog
MODEL_PATH = "model.keras"

EXPERIMENT_NAME = "combined"   # printed in the report, documented in RESULTS.md

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


def compute_class_weights(counts: dict[str, int]) -> dict[int, float]:
    """Make the rare class count for more (proven in experiment/class-weights).

    The training set holds 180 dogs but only 95 cats, so a lazy network can lower
    its loss simply by answering "dog". Weighting each class inversely to how
    often it appears removes that shortcut: getting one cat wrong now costs almost
    twice as much as getting one dog wrong.

    Formula: weight = total_images / (number_of_classes * images_in_this_class).
    The result balances the total cost per class exactly (1.4474 * 95 = 137.5 and
    0.7639 * 180 = 137.5).
    """
    total = sum(counts.values())
    return {
        index: total / (len(CLASS_NAMES) * counts[name])
        for index, name in enumerate(CLASS_NAMES)
    }


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

    This is the SYNTHESIS model: it combines the two architecture-side findings of
    the calibration, each measured on its own branch first.

    * four convolution blocks instead of three (from experiment/deeper-cnn), so
      the network can describe whole shapes such as a head, not just textures,
    * augmentation in front and dropout in the head (from
      experiment/augmentation-dropout), so it cannot memorise its 275 photos.

    The third finding, class weighting, is not a layer - it is applied in main().
    """
    model = keras.Sequential(
        [
            keras.Input(shape=IMG_SIZE + (3,)),

            # --- augmentation (from experiment/augmentation-dropout) -------
            # Every epoch the same photo arrives mirrored and slightly turned,
            # so the model effectively sees far more than 275 examples.
            # Active ONLY while training: Keras switches these off for
            # evaluation and prediction, so predict.py never sees a rotated
            # image and needs no changes.
            layers.RandomFlip("horizontal", seed=SEED),   # a mirrored cat is still a cat
            layers.RandomRotation(0.1, seed=SEED),        # +/- 10% of a turn (~36 deg)

            # Pixels arrive as 0-255 integers; neural nets train far better on
            # small floats, so scale them into the 0-1 range.
            layers.Rescaling(1.0 / 255),

            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            # --- fourth block (from experiment/deeper-cnn) -----------------
            # Also shrinks the tensor reaching Flatten from 16,384 to 8,192
            # values, so the model is deeper yet has fewer parameters.
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(64, activation="relu"),

            # --- dropout (from experiment/augmentation-dropout) ------------
            # 30% of these 64 features are randomly zeroed on every training
            # step, so the verdict cannot depend on one lucky feature.
            layers.Dropout(0.3, seed=SEED),

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
    """Prints one aligned line per epoch instead of Keras' progress bars.

    The "<- saved" marker flags the epochs where val_loss improved, which are
    exactly the epochs ModelCheckpoint writes to disk. It makes the model
    selection visible while training instead of being a silent side effect.
    """

    def on_train_begin(self, logs=None):
        self.best_val_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        marker = ""
        if logs["val_loss"] < self.best_val_loss:
            self.best_val_loss = logs["val_loss"]
            marker = "   <- saved"
        print(
            f"  epoch {epoch + 1:>2}/{EPOCHS}"
            f"   loss {logs['loss']:.4f}  acc {logs['accuracy']:.4f}"
            f"   |   val_loss {logs['val_loss']:.4f}  val_acc {logs['val_accuracy']:.4f}"
            f"{marker}"
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
    """Report the model that was actually saved to disk, not the prettiest epoch."""
    val_loss = history.history["val_loss"]
    val_acc = history.history["val_accuracy"]
    train_acc = history.history["accuracy"]

    # The checkpoint keeps the epoch with the LOWEST val_loss, so that epoch is
    # the model predict.py will load - it is the headline result.
    saved = int(np.argmin(val_loss)) + 1
    # Reported alongside it for honesty: the epoch that "won" on accuracy alone.
    peak = int(np.argmax(val_acc)) + 1

    print(f"\n{LINE}\n  RESULTS ({EXPERIMENT_NAME})\n{LINE}")
    print(f"  majority-class baseline        {baseline_acc:6.2%}"
          f"   (always answer \"{baseline_class}\")")
    print()
    print(f"  saved model -> epoch {saved}"
          f"   (lowest val_loss; this is what predict.py loads)")
    print(f"      val_loss                   {val_loss[saved - 1]:.4f}")
    print(f"      val_accuracy               {val_acc[saved - 1]:6.2%}")
    print(f"      train_accuracy             {train_acc[saved - 1]:6.2%}")
    # Reported in percentage POINTS: the honest way to compare two accuracies.
    print(f"      lift over baseline         "
          f"{(val_acc[saved - 1] - baseline_acc) * 100:+6.2f} pts")
    print()
    print(f"  for reference, the highest val_accuracy of the run was "
          f"{val_acc[peak - 1]:.2%} at epoch {peak}")
    print(f"  (val_loss {val_loss[peak - 1]:.4f} there"
          f"{', so it was not saved' if peak != saved else ''})")
    print(f"{LINE}")
    print(f"  trained in {elapsed:.1f}s on CPU   |   "
          f"epoch {saved} saved -> {MODEL_PATH}")
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
          f"trained from scratch")

    # Class weighting (from experiment/class-weights): applied to the loss, not
    # to the accuracy metric, so the printed accuracies stay comparable with the
    # earlier experiments.
    class_weights = compute_class_weights(train_counts)
    print(f"  weights    "
          + ",   ".join(f"{name} x{class_weights[i]:.2f}"
                        for i, name in enumerate(CLASS_NAMES))
          + "   (inverse class frequency)")
    print(LINE)

    # Keep the weights from the best epoch, not the last one, which may already
    # be overfitting.
    #
    # Why val_loss and not val_accuracy: with only 70 validation images, one image
    # is worth 1.43% of accuracy, so accuracy moves in coarse jumps and a single
    # lucky epoch can win by pure chance. Loss is continuous - it also sees *how
    # confidently* each image was classified - so it is a far less noisy way to
    # pick the epoch worth keeping.
    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_loss", mode="min", save_best_only=True
    )

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, EpochReporter()],
        class_weight=class_weights,   # cats cost more to get wrong
        shuffle=False,   # the tf.data pipeline already reshuffles every epoch
        verbose=0,       # our EpochReporter prints the progress instead
    )
    elapsed = time.time() - start

    print_summary(history, baseline_class, baseline_acc, elapsed)


if __name__ == "__main__":
    main()
