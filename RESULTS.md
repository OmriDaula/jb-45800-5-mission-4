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
| `experiment/more-epochs` | epochs 15 → 30, nothing else | 30 | 68.57% | 74.91% | 3/6 | rejected - more training only memorizes |
| `experiment/deeper-cnn` | one extra Conv2D + MaxPooling block | 15 | – | – | – | pending |
| `experiment/augmentation-dropout` | RandomFlip + RandomRotation + Dropout(0.3) | 25 | – | – | – | pending |
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

### `experiment/more-epochs` - epochs 15 → 30

Doubling the training time changed the result by **nothing at all**: best validation
accuracy was again 68.57% at epoch 6, and not one of the 24 later epochs ever beat
it. Meanwhile training accuracy reached **100.00%** with a training loss of 0.0019,
while validation loss climbed to **1.85** - the model fits its 275 training photos
perfectly and gets steadily *worse* at everything else. The foreign photos still
scored 3/6 with "dog" as the answer every time.

Because the seed is fixed, the first 15 epochs are bit-for-bit identical to the
baseline; the only new information is what happens in epochs 16-30, and the answer
is "nothing good".

**Decision: rejected.** This is the most useful negative result in the log: it proves
the bottleneck is not the amount of training but the lack of data variety and the
class imbalance. Simply waiting longer cannot fix either, so the next experiments
change the model and the training signal instead.
