# 🐱 Cat vs Dog 🐶 — a neural network built from scratch

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange)
![Hardware](https://img.shields.io/badge/hardware-CPU%20only-lightgrey)
![Pretrained](https://img.shields.io/badge/pretrained%20weights-none-brightgreen)

This project teaches a computer to tell **cats from dogs** by looking at photographs.
Nothing is borrowed: the network starts out knowing absolutely nothing — no
pre-trained model, no downloaded weights — and learns only from the 275 photos that
are committed inside this repository. Training takes about a minute and a half on an
ordinary laptop processor, and afterwards the model is asked to judge six photographs
it has never seen in its life. This README tells the honest story of what it learned,
what it failed to learn, and how five controlled experiments proved the difference.

---

## 🚀 Quick start

Everything is committed — the dataset and the test photos included. There are no
manual steps, no downloads, no Docker.

```bash
git clone https://github.com/OmriDaula/jb-45800-5-mission-4.git
cd jb-45800-5-mission-4

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py      # trains the model, creates model.keras (~90 s on CPU)
python predict.py    # classifies all 6 unseen photos in test-images/
```

> **Python version matters.** TensorFlow 2.16.2 is the last release with an Intel-macOS
> wheel and it supports **Python 3.9–3.12 only**. On Python 3.13+ the install will
> fail, so create the virtual environment with an explicit `python3.11` (or `3.12`)
> as shown above.

`predict.py` can also judge a single photo of your own:

```bash
python predict.py my_own_pet.jpg
```

---

## 🎬 Demo

Training ends with a summary that reports the model *actually saved to disk*:

```text
================================================================
  RESULTS (combined)
================================================================
  majority-class baseline        65.71%   (always answer "dog")

  saved model -> epoch 9   (lowest val_loss; this is what predict.py loads)
      val_loss                   0.6007
      val_accuracy               72.86%
      train_accuracy             72.00%
      lift over baseline          +7.14 pts

  for reference, the highest val_accuracy of the run was 75.71% at epoch 3
  (val_loss 0.6869 there, so it was not saved)
================================================================
  trained in 70.5s on CPU   |   epoch 9 saved -> model.keras
  next step: python predict.py
```

Then `predict.py` judges the six foreign photographs:

```text
================================================================
  SUMMARY
================================================================
  image                 prediction  confidence
  -------------------   ----------- ----------
  cat_01_tabby.jpg      dog           69.0%
  cat_02_on_snow.jpg    dog           62.8%
  cat_03_kitten.jpg     dog           81.2%
  dog_01_labrador.jpg   cat           62.7%
  dog_02_beagle.jpg     dog           67.1%
  dog_03_husky.jpg      dog           83.3%
================================================================
```

**Read honestly: 2 of 6.** The model identifies dogs far more reliably than cats
(2 of 3 dogs correct, 0 of 3 cats), and the section
[Honest limits](#-honest-limits--what-the-model-really-learned) explains exactly why —
it is a property of the dataset, not a bug in the code.

---

## 🧠 How it works

A photograph is just a grid of numbers. Each convolution layer looks for patterns and
passes a smaller, more meaningful summary to the next one, until a single number comes
out: the probability that the animal is a dog.

```text
        photo.jpg  (any size, any shape)
             |
             v
    +--------------------+
    |  resize 128 x 128  |
    +--------------------+
             |
             v
    +--------------------+
    |  Rescaling 0..1    |   pixels 0-255  ->  0.0-1.0
    +--------------------+
             |
             v
    +--------------------+
    | Conv2D 16 + Pool   |   learns edges          128 -> 64
    | Conv2D 32 + Pool   |   learns textures        64 -> 32
    | Conv2D 64 + Pool   |   learns parts: an ear   32 -> 16
    | Conv2D 128 + Pool  |   learns whole shapes    16 ->  8
    +--------------------+
             |
             v
    +--------------------+
    | Flatten -> Dense   |   8x8x128 = 8192 numbers -> 64
    | Dropout 0.3        |   forget 30% while training
    +--------------------+
             |
             v
    +--------------------+
    | Dense 1, sigmoid   |   ->  P(dog) = 0.83
    +--------------------+
             |
             v
        "DOG, 83.3% confident"
```

Only **basic building blocks** are used: `Rescaling`, `Conv2D`, `MaxPooling2D`,
`Flatten`, `Dense`, `Dropout`, and the augmentation layers `RandomFlip` /
`RandomRotation`. Total size: **621,857 parameters**, every single one learned from
the 275 training photos.

Two design details prevent the classic beginner bugs:

* **`Rescaling` lives inside the model**, so `predict.py` never has to remember to
  divide by 255 — the preprocessing travels with the model file.
* **`predict.py` reads the input size from the model** (`model.input_shape`) instead
  of hardcoding it, so training and prediction can never disagree about image size.

---

## 🔬 The investigation

The interesting part of this project is not the network — it is how the network was
diagnosed. The first version reached 68.57% validation accuracy, which sounds
respectable until you notice that **always answering "dog" scores 65.71%** on this
data, because dogs outnumber cats almost two to one. The baseline had barely learned
anything at all.

So five experiments were run, each on its own git branch. The first four changed
**exactly one thing** each, so that every result could be attributed to a single
cause; the fifth then combined the changes that had proved themselves.

| # | branch | change | epochs | val_accuracy | train_accuracy | foreign 6 | verdict |
|---|--------|--------|:------:|:------------:|:--------------:|:---------:|---------|
| – | _(no model)_ | always answer "dog" | – | 65.71% | – | 3/6 | the bar to beat |
| 0 | `main` | baseline: 3 conv blocks | 15 | 68.57% | 74.91% | 3/6 | barely above the bar |
| 1 | `experiment/more-epochs` | epochs 15 → 30 | 30 | 68.57% | 74.91% | 3/6 | ❌ rejected |
| 2 | `experiment/deeper-cnn` | +1 Conv2D + MaxPooling block | 15 | 70.00% | 68.73% | 3/6 | ✅ kept as component |
| 3 | `experiment/augmentation-dropout` | RandomFlip + RandomRotation + Dropout | 25 | 68.57% | 72.00% | 3/6 | ✅ kept as component |
| 4 | `experiment/class-weights` | weight classes by inverse frequency | 15 | 71.43% | 64.73% | 2/6 | ✅ kept as component |
| 5 | **`experiment/combined`** | **all three keepers together** | 30 | **72.86%** | **72.00%** | 2/6 | 🏆 **merged into main** |

Three findings were worth more than the accuracy numbers:

**Training longer is not learning more.** Doubling the epochs changed the result by
literally nothing: the best epoch stayed epoch 6, while training accuracy climbed to
100% and validation *loss* nearly tripled (0.63 → 1.85). The network was memorising
its 275 photos, and patience cannot cure that.

**Fixing overfitting did not help — which was the most informative failure.**
Augmentation and dropout worked exactly as designed: training accuracy stopped running
away to 100% and settled around 80%. And the score did not move at all. That proved
memorisation was never the real problem, and pointed at the true culprit — the
imbalanced data — which the class-weight experiment then confirmed.

**A bug hid inside a good-looking number.** The combined model first saved *epoch 3*
with 75.71% validation accuracy, the best figure of the entire project. It was
worthless: training accuracy at that epoch was only 63.64%, and a model cannot be
twelve points better on photos it has never seen than on photos it studied. Running
`predict.py` exposed it instantly — all six answers came back between 50.8% and 51.9%
confidence, a model that had learned to say "maybe" to everything and won on accuracy
by luck. The cause was the selection rule: with 70 validation images, one image is
worth 1.43% of accuracy, so a three-image fluke can crown an epoch. Switching
`ModelCheckpoint` to monitor `val_loss` — continuous, and sensitive to *how
confidently* each photo was judged — fixed it. **The checkpoint criterion turned out
to matter as much as the training itself.**

The full log, including a sixth attempt at 64×64 resolution that was tested and
rejected, is in **[RESULTS.md](RESULTS.md)**.

---

## 📉 Honest limits — what the model really learned

The winning model scores **72.86% validation accuracy** — a genuine +7.14 points over
the always-answer-dog baseline — with a train/validation gap of only **0.86 points**,
the healthiest of every configuration tested. It is not overfitting, and it commits to
real opinions rather than hedging near 50%.

But on the six foreign photographs it gets **2 of 6**, and the pattern is what matters:

| | dogs | cats |
|---|:---:|:---:|
| correctly identified | 2 of 3 | **0 of 3** |

**The model has learned "dog" much better than it has learned "cat", and the reason is
simply the data: there are only 95 cat photos in the training set, against 180 dogs.**
Every fixable cause was in fact fixed during the calibration — memorisation, class
bias, feature depth, epoch selection — and this is what remained. The wall is the size
of the dataset, not the code.

The standard professional solution would be **transfer learning**: start from a network
already trained on millions of photographs. That is deliberately outside the scope of
this assignment, which asks for a network built from basic layers, so the honest
deliverable is a carefully engineered small CNN together with a clear account of where
its limits lie — and the evidence for every claim above.

---

## 📁 Project structure

```text
.
├── train.py           # trains the CNN and saves model.keras          (single file)
├── predict.py         # loads model.keras and classifies photos       (single file)
├── RESULTS.md         # the full calibration log: 5 experiments, all evidence
├── requirements.txt   # pinned, Intel-macOS / CPU compatible
├── data/              # the dataset, committed: 345 images
│   ├── train/         #   cat: 95    dog: 180
│   └── val/           #   cat: 24    dog:  46
└── test-images/       # 6 foreign photos, NOT in the dataset
    ├── cat_01_tabby.jpg  ...
    └── ATTRIBUTION.md    # author + licence for every photo
```

`model.keras` is deliberately **not** committed: it is a generated artefact that
`train.py` recreates in about ninety seconds. This keeps the repository honest — the
results cannot come from a model file nobody can reproduce.

---

## ♻️ Reproducibility

Every run of `train.py` produces **identical numbers**. This was verified the strict
way: the repository was cloned fresh into an empty directory, a new virtual environment
was built, and the whole flow was re-run. It selected the same epoch (9), the same
`val_loss` (0.6007) to four decimal places, and produced the same six predictions to a
tenth of a percent. What makes that possible:

* one fixed seed (`SEED = 42`) for Python, NumPy and TensorFlow, set through
  `keras.utils.set_random_seed`,
* `tf.config.experimental.enable_op_determinism()`, so even the low-level TensorFlow
  kernels cannot introduce drift,
* pinned dependency versions in `requirements.txt`,
* the saved epoch is chosen by a deterministic rule (lowest `val_loss`) and printed
  explicitly, so the reported numbers always describe the model that is on disk.

---

## 🖼️ Test images and licences

The six foreign photographs come from Wikimedia Commons and were verified to be
byte-for-byte different from all 345 dataset images (compared by MD5 hash). Author and
licence for each one are credited in
[`test-images/ATTRIBUTION.md`](test-images/ATTRIBUTION.md). Their filenames state the
true animal, so any prediction can be checked at a glance.

---

## 📚 Dataset

[**marquis03/cats-and-dogs**](https://www.kaggle.com/datasets/marquis03/cats-and-dogs)
from Kaggle (Apache-2.0), 345 images in total, ~10 MB — small enough to commit, which
is what makes the one-command reproduction in *Quick start* possible.
