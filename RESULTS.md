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
| `experiment/augmentation-dropout` | RandomFlip + RandomRotation + Dropout(0.3) | 25 | – | – | – | pending |
| `experiment/class-weights` | `class_weight` inversely proportional to class frequency | 15 | 71.43% | 64.73% | 2/6 | bias removed, exposing weak features - essential component, not sufficient alone |
| `experiment/combined` | synthesis: 4 conv blocks + augmentation + dropout + class weights | 30 | – | – | – | pending |

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

### `experiment/class-weights` - inverse-frequency class weighting

`model.fit` now receives `class_weight={0: 1.4474, 1: 0.7639}`, computed from the
real counts as `total / (n_classes * count_of_this_class)`. A cat mistake costs
**1.9x** more than a dog mistake, and the total weighted cost per class comes out
exactly equal (`1.4474 x 95 = 137.5` and `0.7639 x 180 = 137.5`), so the network
trains as if the dataset were balanced. Only the loss is weighted - Keras' accuracy
metric still counts every image equally, so these numbers stay comparable with the
other rows.

Best validation accuracy was **71.43%** (epoch 5), the highest of all four
experiments, achieved while training accuracy was only 64.73% - the model was still
generalising, not memorising.

The foreign photos scored **2/6**, the *lowest* of the project, and that number is
the most informative result in the whole log. The predictions no longer look
anything like before:

| | baseline / deeper / augmented | class-weights |
|---|---|---|
| confidence range | 69% - 92% | **51% - 58%** |
| answers | "dog" six times out of six | mixed |

The majority-class bias is **gone**: the model no longer leans on "dog" as a safe
default. But with the bias removed, what is left underneath is a model sitting near
50/50 on every photo - it has almost no genuinely discriminative features. The
labrador landed on the wrong side of 0.5 essentially by chance, which is why 2/6 is
not really worse than 3/6; both are noise around a coin flip.

**Decision: essential component, not sufficient alone.** Class weighting removes the
shortcut that hid the real weakness. Now that the model can no longer cheat, it needs
better features and more input variety to actually learn - which is what the final
synthesis experiment provides.

## Synthesis phase

Experiments 1-4 were deliberately **single-variable**: each changed exactly one thing
so that its effect could be attributed with confidence. That isolation phase produced
three separate findings:

* depth improved features and closed the train/validation gap (`deeper-cnn`),
* augmentation and dropout genuinely stopped memorisation (`augmentation-dropout`),
* class weighting removed the majority-class bias (`class-weights`),

and one clear negative result (`more-epochs`: more training changes nothing).

No single change was enough, because the baseline had **three independent problems**.
`experiment/combined` therefore stops isolating and starts synthesising: it applies
every proven component at once, which is a legitimate final step precisely *because*
each part was measured on its own first.
