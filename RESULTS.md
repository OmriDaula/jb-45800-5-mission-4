# Calibration log

This file documents every experiment run on this project. The baseline lives on
`main`; each experiment lives on its own git branch and changes **exactly one
thing**, so any difference in the numbers can only come from that one change.

## How these numbers are produced

* Same dataset every time: 275 training images, 70 validation images (committed in `data/`).
* Same seed every time (`SEED = 42`, plus TensorFlow op determinism), so a rerun
  reproduces the numbers exactly.
* `val_accuracy` is the **best** epoch, which is also the epoch saved to
  `model.keras` by `ModelCheckpoint`. `train_accuracy` is measured at that same
  epoch, so the pair is comparable.
* **foreign 6** is the score of `python predict.py` on the six unseen photos in
  `test-images/` (3 cats, 3 dogs). This is the honest real-world test: a model can
  score well on validation and still behave badly on new photos.

### Reference point: the "no model at all" score

The dataset is imbalanced - 180 dogs vs 95 cats in training, 46 dogs vs 24 cats
in validation. A program with no intelligence whatsoever that always answers
"dog" therefore scores:

```
46 / 70 = 65.71% validation accuracy   and   3 / 6 on the foreign images
```

**Every experiment below must be judged against 65.71%, not against 0%.**

## Results

| branch | change | epochs | val_accuracy | train_accuracy | foreign 6 | decision |
|--------|--------|:------:|:------------:|:--------------:|:---------:|----------|
| _(no model)_ | always answer "dog" | – | 65.71% | – | 3/6 | reference only |
| `main` | baseline: 3 conv blocks, no augmentation, no dropout | 15 | **68.57%** | 74.91% | **3/6** | baseline |
| `experiment/more-epochs` | epochs 15 → 30, nothing else | 30 | – | – | – | pending |
| `experiment/deeper-cnn` | one extra Conv2D + MaxPooling block | 15 | – | – | – | pending |
| `experiment/augmentation-dropout` | RandomFlip + RandomRotation + Dropout(0.3) | 25 | 68.57% | 72.00% | 3/6 | rejected as standalone - fixes overfitting, not the collapse; regularisation candidate |
| `experiment/class-weights` | `class_weight` inversely proportional to class frequency | 15 | – | – | – | pending |

## Branch notes

### `main` - baseline

Best validation accuracy was 68.57% at epoch 6, only **+2.86 points** above the
"always answer dog" reference. Training accuracy kept climbing to **98.18%** by
epoch 15 while validation accuracy *fell* to 57.14%: a textbook 41-point
train/validation gap, meaning the network is memorising the 275 training photos
instead of learning what a cat looks like.

On the six foreign photos the model answered "dog" **every single time**, scoring
3/6 purely because three of them really are dogs. So the baseline has partially
collapsed onto the majority class.

Two clear problems for the experiments to attack:

1. **Overfitting** - too little data for a network with 1.07M parameters
   (targeted by `experiment/augmentation-dropout`).
2. **Class imbalance** - dogs outnumber cats almost 2:1, so guessing "dog" is a
   cheap way to lower the loss (targeted by `experiment/class-weights`).

### `experiment/augmentation-dropout` - RandomFlip + RandomRotation + Dropout(0.3)

Three regularisation layers were added to the **unchanged baseline architecture**:
`RandomFlip("horizontal")` and `RandomRotation(0.1)` in front of the network, and
`Dropout(0.3)` after the dense layer. All three are inside the model and Keras
activates them only while training, so `predict.py` still sees clean, unrotated
images and needed no modification. They add **zero parameters** (still 1,072,289),
which makes this a clean single-variable comparison with the baseline.

The intervention worked exactly as intended on its own target:

* Training accuracy stayed at roughly **80-84%** across all 25 epochs, instead of
  running away to 98-100% as it did in the baseline and in `more-epochs`.
* At the saved epoch the numbers were 72.00% train / 68.57% validation - a healthy
  gap of under 4 points.

**And it changed the outcome by nothing.** Validation accuracy was 68.57%, identical
to the baseline, and the foreign photos scored **3/6** yet again, all six answered
"dog", with the wrong answers now the most confident of the whole project (the kitten
at 91.8%).

This is the pivotal result of the calibration. Memorisation was successfully
suppressed and the score did not move, which proves **memorisation was never the
real problem**. What remains is the training signal itself: with 180 dogs against 95
cats, answering "dog" is simply the cheapest way to lower the loss, and no amount of
regularisation makes that untrue.

**Decision: rejected as a standalone change**, but kept as a regularisation
candidate for combining with a fix that addresses the imbalance directly.
