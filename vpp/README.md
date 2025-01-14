# vpp [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Model description

The base networks of the model were Inception-v3, Inception-v4, and Xception, all pre-trained on ImageNet. For the competition classification task, the following modifications were made:

1. Changed the number of input channels of first convolution to 4 (red, green, blue, yellow).
2. Changed the last pooling layer to global average pooling.
3. Appended a fully connected layer with output dimension 128 after the global pooling.
4. Appended a batch normalization layer and a dropout layer before the additional fully connected layer.

We used both official (both PNG and TIFF) datasets and the HPAv18 external data. Three different scales were used during training (512 ⨉ 512 for PNG images and 650 ⨉ 650, 800⨉800 for TIFF images). The HPA dataset has four channels each of which is an RGB image of its own, so we took only one channel (r=Hrr, g=Hgg, b=Hbb, y=Hyb, where Hyb is the blue channel of the yellow dyeing mode image from HPAv18 dataset) to form a 4-channel (r,g,b,y) input for training. Rotation, flip, and shear were very effective augmentations to increase the amount of training data.

To handle the class imbalance, different sampling weights were set for different classes, to ensure that categories with fewer labels have a higher probability of sampling. A multi-label one-versus-all loss based on max-entropy (MultiLabelSoftMarginLoss) was used for all the models across 10-fold cross validation sets with 8% for validation. All the models were trained with stochastic gradient descent (SGD) with momentum set to 0.9, and weight decay of 1e-4. The initial learning rate for input size 512 ⨉ 512 was set to 0.05, and 0.01 for 650 ⨉ 650 and 800 ⨉ 800. A step learning rate scheduler with gamma of 0.1 and step of 6 was applied. The training process was divided into two stages, where the first stage used 512 ⨉ 512 with models trained on ImageNet, and the second stage used 650 ⨉ 650 or 800 ⨉ 800 with model trained from the first stage.

The features of Inceptionv3, Inceptionv4 and Xception with 10 folds and size 800 ⨉ 800 were extracted, and then concatenated into a new feature vector with 3840 dimensions. A sample multi-layer perceptron network was created to classify 28 categories from the new features, with three fully connected layers (3840, 512, and 28 neurons respectively). Two loss functions were tested: MultiLabelSoftMarginLoss and BCEWithLogitsLoss, which reached scores of 0.5515/0.62791 and 0.55227/0.62963 respectively.

The final model is an ensemble of the above methods.

## Model source

The original model source can be found [here](https://github.com/CellProfiling/HPA-competition-solutions/tree/master/vpp).

## Trained model files

The trained model files can be found [here](https://kth.box.com/s/12k7y6xf0xfqsqoq6op4ihic88ddhali).

## Model usage

We provide two different versions as follows:

- Tiny ([run_submit_tiny.sh](https://github.com/CellProfiling/HPA-competition-solutions/tree/master/vpp/src/run_submit_tiny.sh)) : It's a simple version which got 0.57196/0.64658 on the leaderboard with a small number of models.
- Final ([run_submit_final.sh](https://github.com/CellProfiling/HPA-competition-solutions/tree/master/vpp/src/run_submit_final.sh)) : It's our final submission which got 0.56689/0.64615 on the leaderboard.

### 1. Image Preprocessing

- All PNG images are kept at their original 512 size, whereas the TIFF images are resized to 1024 (.png).
- Modified DATASET_PATH_512(original,512x512) and DATASET_PATH_1024(png,tiff resized to 1024x1024) at [config.py](https://github.com/CellProfiling/HPA-competition-solutions/tree/master/vpp/src/config.py):

### 2. Run run_summit_.sh

We uploaded some of models, which can got 0.57254 on the private leaderboard. ([submission file](https://github.com/CellProfiling/HPA-competition-solutions/tree/master/vpp/src/results/final_submission_multhr.csv))

You can also run the following script to get the submission file:

```sh
chmod +x run_submit_*.sh
./run_submit_.sh
```
