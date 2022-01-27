# AutoSet
## **AutoSet** is a computer vision program that plays the card game Set® (a pattern recognition game)

Read the rules of Set® [here](https://en.wikipedia.org/wiki/Set_(card_game)).

## Features:
-   custom-trained convolutional neural network models that classify each visible card according to its number, color, shade, and shape
-   an algorithm to identify sets among the visible cards
-   a heads-up display to show the found sets to the operator

## Detection Models
Four separate classification [models](detection_models/trained models/) for classifying a card's number, shade, color, and shape trained using the included [training images](/model training dataset).
Each model is approximately 98% accurate and was trained using [Tensorflow](https://www.tensorflow.org/).
Please consult the [CNN model trainer](detection_models/CNN_model_trainer.py) to view the model architecture and other training parameters.

## Training Images
Includes 486 RGB images of single cards (six samples of each of 81 unique cards) with varying exposure, color temperature, and orientation.
The [training_labels.csv](training_labels.csv) spreadsheet contains class labels for each image in the training dataset.

## Heads-Up Display
Displays an image of the visible cards with color-matching bounding boxes drawn around cards in a set.

![heads-up display demo reduced](https://user-images.githubusercontent.com/97372919/151372168-b1d62699-9dac-4ad0-9812-9dbdf874b0c9.jpg)


## Known Issues
The probability of accurately classifying the four attributes for each of 12 cards in a standard round of Set is approximately 0.38 [0.98 model accuracy ^ (4 attr. * 12 cards)].
With this accuracy, many of the sets found by the program are likely to be incorrect.  Ideally, the classification models would have an accuracies closer to 100%.


## License

[The GNU General Public License v3.0](LICENSE) Copyright © 2021 Peter Reynolds

