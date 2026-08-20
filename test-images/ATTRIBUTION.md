# Foreign test images - sources and licenses

These six photos are **not part of the training data**. They were downloaded from
Wikimedia Commons specifically to test the trained model on unseen input, and
verified to be byte-for-byte different from all 345 images in `data/`.

The filename states the true animal, so the prediction can be checked at a glance.

| File | True label | Author | License | Source |
|------|-----------|--------|---------|--------|
| `cat_01_tabby.jpg` | cat | Alvesgaspar | CC BY-SA 3.0 | [Commons](https://commons.wikimedia.org/wiki/File:Cat_November_2010-1a.jpg) |
| `cat_02_on_snow.jpg` | cat | Von.grzanka | CC BY-SA 3.0 | [Commons](https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg) |
| `cat_03_kitten.jpg` | cat | André Karwath (Aka) | CC BY-SA 2.5 | [Commons](https://commons.wikimedia.org/wiki/File:Six_weeks_old_cat_(aka).jpg) |
| `dog_01_labrador.jpg` | dog | Djmirko (derivative work) | CC BY-SA 3.0 | [Commons](https://commons.wikimedia.org/wiki/File:YellowLabradorLooking_new.jpg) |
| `dog_02_beagle.jpg` | dog | Wikimedia Commons contributor | CC BY-SA 3.0 | [Commons](https://commons.wikimedia.org/wiki/File:Beagle_600.jpg) |
| `dog_03_husky.jpg` | dog | Per Harald Olsen | CC BY 2.5 | [Commons](https://commons.wikimedia.org/wiki/File:Siberian_Husky_pho.jpg) |

All images were resized to a maximum width of 640-960 px to keep this repository
small. Each license requires attribution, which this file provides; follow the
Source link for the full license text and the original file.
