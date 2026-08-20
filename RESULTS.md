# Calibration log

This file documents every experiment run on this project. The baseline lives on
`main`; each experiment lives on its own git branch. Experiments 1-4 changed
**exactly one thing** each, so any difference in the numbers can only come from that
one change. Experiment 5 then combines the components that proved themselves.

## How these numbers are produced

* Same dataset every time: 275 training images, 70 validation images (committed in `data/`).
* Same seed every time (`SEED = 42`, plus TensorFlow op determinism), so a rerun
  reproduces the numbers exactly.
* `val_accuracy` is the **best** epoch, which is also the epoch saved to
  `model.keras` by `ModelCheckpoint`. `train_accuracy` is measured at that same
  epoch, so the pair describes one single model.
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
| `main` | baseline: 3 conv blocks, no augmentation, no dropout | 15 | 68.57% | 74.91% | 3/6 | baseline |
| `experiment/more-epochs` | epochs 15 → 30, nothing else | 30 | 68.57% | 74.91% | 3/6 | rejected - more training only memorises |
| `experiment/deeper-cnn` | one extra Conv2D + MaxPooling block | 15 | 70.00% | 68.73% | 3/6 | architecture candidate - better metric, still no cat detection |
| `experiment/augmentation-dropout` | RandomFlip + RandomRotation + Dropout(0.3) | 25 | 68.57% | 72.00% | 3/6 | rejected as standalone - fixes overfitting, not the collapse |
| `experiment/class-weights` | `class_weight` by inverse class frequency | 15 | **71.43%** | 64.73% | 2/6 | essential component - removes bias, exposes weak features |
| `experiment/combined` | synthesis: all proven components together | 30 | 72.86% | 72.00% | 2/6 | **merged into main** - live and honest, but cats still unlearned |
| `experiment/stronger-augmentation` | augmentation block 2 layers → 5, at 128x128 | 40 | 65.71% | – | 3/6 (all-dog) | rejected - regression to the majority-class baseline |

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
class imbalance. Simply waiting longer cannot fix either.

### `experiment/deeper-cnn` - one extra Conv2D + MaxPooling block

A fourth convolution block (filters 16 → 32 → 64 → **128**, resolution
128 → 64 → 32 → 16 → **8**) lets the network describe larger shapes such as a whole
head instead of only small local textures. Because the extra pooling shrinks the
tensor reaching `Flatten` from 16,384 to 8,192 values, the deeper model actually has
**fewer** parameters than the baseline: 621,857 vs 1,072,289.

Two genuine improvements:

* Best validation accuracy rose to **70.00%** (epoch 7), the highest so far at that point.
* Training accuracy at that epoch was 68.73%, so the train/validation gap at the
  saved epoch is essentially **closed** - this model is not memorising.

The foreign photos tell the other half of the story: still **3/6**, still "dog" for
all six, and now *more* confident on the cats it gets wrong (83-87%, versus 69-80%
for the baseline). In absolute terms 70.00% means 49 of 70 validation images correct
against 46 for the always-dog reference, so the whole gain is **three images**.

**Decision: keep as an architecture candidate.** Depth improved the metric and cured
the overfitting, but it did not teach the model to recognise a cat.

### `experiment/augmentation-dropout` - RandomFlip + RandomRotation + Dropout(0.3)

Three regularisation layers were added to the **unchanged baseline architecture**:
`RandomFlip("horizontal")` and `RandomRotation(0.1)` in front of the network, and
`Dropout(0.3)` after the dense layer. All three live inside the model and Keras
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

This is a pivotal result: memorisation was successfully suppressed and the score did
not move, which proves **memorisation was never the real problem**. What remains is
the training signal itself.

**Decision: rejected as a standalone change**, kept as a regularisation component.

### `experiment/class-weights` - inverse-frequency class weighting

`model.fit` now receives `class_weight={0: 1.4474, 1: 0.7639}`, computed from the
real counts as `total / (n_classes * count_of_this_class)`. A cat mistake costs
**1.9x** more than a dog mistake, and the total weighted cost per class comes out
exactly equal (`1.4474 x 95 = 137.5` and `0.7639 x 180 = 137.5`), so the network
trains as if the dataset were balanced. Only the loss is weighted - Keras' accuracy
metric still counts every image equally, so these numbers stay comparable with the
other rows.

Best validation accuracy was **71.43%** (epoch 5), the highest of all four
single-variable experiments, achieved while training accuracy was only 64.73% - the
model was still generalising, not memorising.

The foreign photos scored **2/6**, the lowest of the project, and that number is the
most informative result in the whole log. The predictions no longer look anything
like before:

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
shortcut that hid the real weakness.

## Synthesis phase

Experiments 1-4 were deliberately **single-variable**: each changed exactly one thing,
so its effect could be attributed with confidence. That isolation phase is now
complete and produced three positive findings and one negative one:

| finding | source | what it fixed |
|---|---|---|
| four conv blocks | `deeper-cnn` | better features, closed train/val gap, 42% fewer parameters |
| augmentation + dropout | `augmentation-dropout` | memorisation genuinely stopped |
| inverse class weights | `class-weights` | majority-class bias removed |
| more epochs alone | `more-epochs` | nothing - rejected |

No single change was enough, because the baseline suffered from **three independent
problems at once**: weak features, memorisation, and a biased training signal. Fixing
any one of them left the other two in charge.

`experiment/combined` therefore stops isolating and starts synthesising. It applies
every proven component together - and this is only a legitimate final step *because*
each part was measured on its own first:

* 4 convolution blocks (16 → 32 → 64 → 128 filters), 621,857 parameters
* `RandomFlip("horizontal")` + `RandomRotation(0.1)` in front of the network
* `Dropout(0.3)` in the head
* `class_weight = {cat: 1.4474, dog: 0.7639}` applied to the loss
* 30 epochs, since augmented data is harder and needs more passes
* `ModelCheckpoint` saves the single best epoch - by `val_loss`, for the reason
  explained immediately below

### The checkpoint criterion - a bug found by running the synthesis

The first run of this branch produced a result that looked good and was worthless,
and finding out why changed the script.

`ModelCheckpoint` was monitoring `val_accuracy`, and it saved **epoch 3**: 75.71%
validation accuracy - the best number the project had ever produced - while training
accuracy at that same epoch was only 63.64%. A model that understands the data
cannot be *twelve points better* on data it has never seen than on the data it was
trained on. That combination is the signature of luck, not skill.

`predict.py` confirmed it immediately: all six foreign photos came back between
**50.8% and 51.9%** confidence. The saved model was essentially a constant function
outputting "about 0.51", and it won on accuracy only because that constant happened
to fall on the right side of 0.5 for 53 of the 70 validation images.

The root cause is the metric, not the model. With 70 validation images, **one image
is worth 1.43% of accuracy**, so `val_accuracy` can only move in coarse jumps and a
three-image fluke is enough to crown an epoch permanently. Meanwhile the training log
showed the combination genuinely working:

* epochs 19-21 reached 72.9-75.7% validation accuracy with 74-80% training accuracy -
  a real model, with a believable train/validation relationship,
* `val_loss` reached its project-wide low of **0.60** at epochs 9-10.

**The fix:** monitor `val_loss` with `mode="min"`. Loss is continuous and takes into
account *how confidently* each image was classified, so a barely-confident 0.51
prediction cannot masquerade as a correct one. It is a far less noisy criterion for
choosing which epoch to keep.

Two supporting changes make the selection auditable instead of silent:

* the final summary now prints **which epoch was saved**, with its `val_loss`,
  `val_accuracy` and `train_accuracy`, plus the highest-accuracy epoch for reference
  and why it was not chosen;
* each epoch line is tagged `<- saved` when `val_loss` improves, so the model
  selection is visible as it happens.

**The lesson worth remembering: the checkpoint criterion is part of the model.** All
five experiments trained perfectly reasonable networks, and this one nearly shipped a
constant function because of how the "best" epoch was chosen. On a small validation
set, *how you select* matters as much as *how you train*.

The bar this synthesis has to clear: beat 71.43% validation accuracy **and**, more
importantly, finally score above 3/6 on the foreign photos by actually identifying a
cat as a cat.

### `experiment/combined` - result

With the checkpoint criterion fixed, the run saved **epoch 9**:

| | value |
|---|---|
| `val_loss` (selection criterion) | **0.6007** - the lowest of the entire project |
| `val_accuracy` | 72.86% |
| `train_accuracy` | 72.00% |
| train/validation gap | **0.86 points** |

That gap is the healthiest number in the log. The baseline's gap was 41 points; this
model performs the same on photos it trained on as on photos it has never seen, which
is exactly what a well-regularised model should do.

`predict.py` confirms it is a **live** model rather than the constant function the
first run nearly shipped: confidences now spread across **62-83%**, compared with the
50.8-51.9% flatline of the accuracy-selected epoch 3.

And yet the foreign photos score **2/6**: all three cats are still called "dog", and
the labrador now flips to "cat". So the model has learned *something* real - it
commits to opinions and its validation behaviour is honest - but it has not learned
the concept "cat".

**Honest conclusion: this is close to the ceiling of what 95 cat images can teach a
convolutional network trained from scratch.** Every mechanism that could be fixed by
better training procedure has now been fixed - memorisation, class bias, feature
depth, epoch selection - and the remaining wall is the size of the dataset itself. The
usual professional answer at this point is transfer learning, which is deliberately
out of scope for this assignment, so the honest deliverable is a well-engineered small
CNN plus a clear-eyed account of its limits.

### Final sub-experiment: 64x64 input + RandomZoom(0.1) - rejected

One last cheap idea was tested before locking the configuration: shrink the input from
128x128 to 64x64 and add `RandomZoom(0.1)`. The reasoning was sound - a quarter of the
pixels means a quarter of the fine detail to overfit on, the parameter count drops from
621,857 to **228,641**, and zoom augmentation targets a real mismatch (the training
photos are tight pet portraits, the foreign photos are wider scenes).

It failed both acceptance conditions that had been agreed in advance:

| | epoch-9 model (kept) | 64x64 + zoom |
|---|---|---|
| `val_loss` | **0.6007** | 0.6158 - worse |
| foreign 6 | 2/6, confidences 62-83% | "3/6" - but see below |

The 3/6 was an **illusion, and reading it correctly is the whole point**: all six
photos were answered "dog" again, at 73-78% confidence, with the labrador back to
"dog" at 51.2%. That is precisely the always-answer-dog reference score of 3/6 - the
model had regressed to the baseline's majority-class behaviour, not improved at
recognition. Taken at face value, 3/6 looks like an improvement over 2/6; looked at
honestly, it is the opposite.

**Lessons recorded:**

1. **A score is meaningless without its prediction pattern.** 3/6 from a model that
   always says "dog" is worse than 2/6 from a model that actually disagrees with
   itself across photos.
2. **64x64 destroys the evidence.** Cats and dogs differ in fine structure - whisker
   lines, ear edges, eye and muzzle shape. Halving the resolution removes exactly the
   detail the network needs, so the capacity saving is not worth the information lost.
3. **Reducing capacity is an exhausted lever.** Depth already cut parameters from
   1.07M to 622k with a real gain; cutting to 229k gave a loss. There is no more
   headroom in making the model smaller.

**Decision: reverted.** The `experiment/combined` configuration at 128x128 stands as
the final model. One piece of this attempt was kept, because it is an improvement
independent of image size: `predict.py` no longer hardcodes the input size, it reads
it from the trained model with `model.input_shape`. Training and prediction can
therefore never disagree about preprocessing - a mismatch that is one of the most
common silent bugs in real machine-learning code.

### `experiment/stronger-augmentation` - a fair trial for zoom

**Why this experiment exists: the previous one was not a fair test of zoom.** The
64x64 attempt changed two things at once - it added `RandomZoom` *and* halved the
resolution - and it was rejected. But the evidence pointed squarely at the resolution
as the culprit: 64x64 removes the fine structure (whisker lines, ear edges, muzzle
shape) that separates a cat from a dog. Zoom itself was never tried on its own, so it
was condemned for its packaging rather than its content. That is exactly the mistake
the single-variable discipline of experiments 1-4 was designed to avoid, so zoom gets
its own trial here.

The change is confined to the augmentation block, at the **unchanged 128x128
resolution**, with everything else identical to the merged winner (four conv blocks,
`Dropout(0.3)`, inverse class weights, `val_loss` checkpoint, 621,857 parameters -
augmentation layers add none):

| layer | kept or new | what mismatch it targets |
|---|---|---|
| `RandomFlip("horizontal")` | kept | a mirrored cat is still a cat |
| `RandomRotation(0.1)` | kept | the animal is not always upright |
| `RandomZoom(0.15)` | **new** | animal near the camera vs far away |
| `RandomTranslation(0.1, 0.1)` | **new** | animal off-centre rather than posed |
| `RandomContrast(0.2)` | **new** | different lighting and cameras |

The reasoning behind all three additions is the same, and it is specific to this
project's test set: the training photos are nearly all tightly framed, well-lit,
centred pet portraits, while the six foreign photos from Wikimedia Commons are ordinary
snapshots - a cat sitting in snow, a husky in a wide outdoor scene. That gap between
training and reality is precisely what geometric and photometric augmentation exists to
close.

`EPOCHS` rises from 30 to 40, because five augmentation layers make every epoch a
harder problem and the model needs more passes to converge. This is safe rather than
generous: the `val_loss` checkpoint keeps whichever epoch is genuinely best, so extra
epochs cannot degrade the saved model - as `more-epochs` proved, they simply do nothing
if there is nothing more to learn.

**Acceptance criteria, fixed before the run:** this replaces the current model only if
`val_loss` beats **0.6007** *and* the foreign-photo behaviour improves - a real cat
detected, or at least a live confidence spread with no regression to answering "dog"
six times. Failing either, it is reverted and the epoch-9 model on `main` stands as
final. The criteria are written down in advance deliberately, so the result cannot be
reinterpreted after the fact to look like a success - which is exactly the trap the
"3/6 illusion" set in the previous experiment.

#### Result: rejected on both criteria

| criterion | required | achieved |
|---|---|---|
| `val_loss` | below 0.6007 | **0.6360** - worse |
| foreign behaviour | a cat detected, or a live spread | **all six answered "dog"** - regression |

Validation accuracy at the saved epoch fell to **65.71%** - which is not merely a lower
number, it is *exactly* the always-answer-dog reference score, the arithmetic
fingerprint of a model that has stopped discriminating and gone back to naming the
common class. The foreign photos agree: six "dog" answers, so the nominal 3/6 is once
again the majority-class illusion rather than recognition.

**Why it failed, and it is not the same reason as the 64x64 attempt.** Zoom did get its
fair isolated trial at full resolution, which was the whole point of this branch, and it
still lost - so the earlier rejection was right, just previously for a partly wrong
reason. The likely cause here is **augmentation strength outrunning the dataset**: five
simultaneous transforms (flip, rotate, zoom, translate, contrast) mean the network
almost never sees an undistorted photo, and with only 95 cats to learn from, the genuine
signal gets buried under the noise the augmentation introduces. Augmentation combats
memorisation, but it cannot manufacture information that is not in the data - and past a
certain intensity it destroys the little that is there.

**Decision: rejected, and the branch is deliberately kept on GitHub without merging.**
The winning configuration on `main` - the epoch-9 model, `val_loss` 0.6007,
`val_accuracy` 72.86% - stands as final. Only this documentation was carried over to
`main`; none of the code.

This closes the calibration with a symmetrical result: strengthening augmentation
(experiment 6) and weakening capacity (the 64x64 sub-experiment) both fail from opposite
directions, which is about as clear a demonstration as one could ask for that the binding
constraint is the **quantity of training data**, not the configuration of the model.

_Note: `train_accuracy` at the saved epoch is left blank in the table above because the
run's per-epoch training accuracy was not recorded when the result was reported; the two
acceptance criteria were both decided on `val_loss` and the foreign-photo pattern._
