# Standford Cars  - Image Classification

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/morganmcg1/Projects/master)
  <- Launch Binder or share the [Binder link](https://mybinder.org/v2/gh/morganmcg1/Projects/master)

Image classification of the stanford-cars dataset leveraging the fastai v1. The goal is to **try hit 90%+ accuracy**, starting with a basic fastai image classification workflow and interating from there. My 90%+ goal is based on @sgugger's code implementing Adam for the Stanford Cars dataset, here: https://github.com/sgugger/Adam-experiments

This was all run on a Paperspace P4000 machine.

**labels_df.csv** contains the labels, filepath and test/train flag for each image file.

## Notebook Results

**1_stanford_cars_basic.ipynb**

 - Benchmark model using basic fastai image classification workflow including the 1-cycle policy
 - **84.95%** Accuracy
 
 **2_stanford_cars_lr_tuning.ipynb**

 - Tuning of the learning rate and differential learning rates, again using fastai's implementation of the 1-cycle policy
 - **88.19%** Accuracy, up 3.2%
 
 **3_stanford_cars_cropped.ipynb**

 - Training the model using the cropped images, based on the bounding boxes provided
 - **78.54%** Accuracy, down 9.5% from Notebook 2 
 
 **4_stanford_cars_mixup.ipynb**

 - Tuning the model using the [Mixup](https://arxiv.org/abs/1710.09412)) protocol, blending input images to provide stronger regularisation
 - **89.4%** Accuracy, up 1% since NB2
 
  **5_stanford_cars_mixup_and_dropout.ipynb**

 - Tuning the dropout parameters while also using the [Mixup](https://arxiv.org/abs/1710.09412)) protocol
 - **89.2%** Accuracy achieved with agressive dropout (ps = [0.35, 0.7]), accuracy more or less the same as NB4
 
 **6_stanford_cars_cutout.ipynb**
 - Used the Cuout data augmentation alongside default fastai data transforms, size of the squares were 25% of the image side (e.g. 25%  x 224)
 - **88.3%** Accuracy achieved

## S0TA 
- **95%** - WS-DAN - [See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification - Hu 2019](https://arxiv.org/abs/1901.09891). Code might not be released until October 2019 if it is accepted for ICCV-2019.
- Previous SOTA - **93.61%** (Apr-18)  https://www.researchgate.net/publication/316027349_Deep_CNNs_With_Spatially_Weighted_Pooling_for_Fine-Grained_Car_Recognition

## Potential Avenues of Investigation
FORNAX - Great roundup in advances in 2018, some of which can be applied: https://github.com/kmkolasinski/deep-learning-notes/blob/master/seminars/2018-12-Improving-DL-with-tricks/Improving_deep_learning_models_with_bag_of_tricks.pdf

AMAZON - Bag of Tricks for Image Classification with Convolutional Neural Network: https://arxiv.org/pdf/1812.01187.pdf

Multi-Attention CNN: https://github.com/Jianlong-Fu/Multi-Attention-CNN

- Visualise with images in t-sne: https://github.com/kheyer/ML-DL-Projects

#### Data Augmentation
- Great visualisaton here  for the [transforms available in fastai](https://www.kaggle.com/init27/introduction-to-image-augmentation-using-fastai)
- Train only on cropped images
- Use Mixup 
    - Paper: https://arxiv.org/abs/1710.09412
    - Paper repo: https://github.com/facebookresearch/mixup-cifar10
    - Fastai docs: https://docs.fast.ai/callbacks.mixup.html , https://forums.fast.ai/t/mixup-data-augmentation/22764/21)
- Mixup + Dropout (produced good results in Mixup paper)
- AdaMixup (https://arxiv.org/abs/1809.02499v3)
- Cutout - Improved Regularization of Convolutional Neural Networks with Cutout https://arxiv.org/pdf/1708.04552.pdf
    - https://docs.fast.ai/vision.transform.html#_cutout
- RGB Transforms, which I tested [here](https://github.com/morganmcg1/Projects/blob/master/feature-testing/RGB%20Transformation%20Testing.ipynb)
- Label Smoothing (https://arxiv.org/abs/1512.00567)
    - Mixup + Label smoothing (tested in Mixup paper) - maybe not, didn't produce great results
- Random Erasing (Zhong et al., 2017)
    - Paper https://arxiv.org/abs/1708.04896 
- Try increase zoom and higher resolution images
- Use own stats (mean+std dev) from training set to normalize the images
- Use non-standard fastai image augmentations, including augmentations for this dataset can be found here: http://ee.sharif.edu/~shayan_f/fgcc/index.html 

#### Training Regimes
- Agressive LR for training all layers
- Adding Weight-Decay and tuning Dropout
- Batch Norm
- @sgugger's adam experiments: https://github.com/sgugger/Adam-experiments
- AdamW with 1-cycle: https://twitter.com/radekosmulski/status/1014964816337952770?s=12
- AdamW and other DL tricks: https://twitter.com/drsxr/status/1073208269907353602?s=12
- train with bn_freeze=true for unfrozen layers
- Shake-Shake Regularisation (mentioned in Fornax slides above)
- Knowledge Distillation (paper: https://arxiv.org/abs/1503.02531, https://forums.fast.ai/t/part-1-complete-collection-of-video-timelines/5504)
- Kaggle winner, Mixup + Knowledge Distillation (http://arxiv.org/abs/1809.04403v2)

#### Architecture
- Try alternate resnet sizes (benchmark used resnet152)
- Try alternate archs, e.g. densenet, unet
- [Try XResnet152](https://twitter.com/jeremyphoward/status/1115036889818341376?s=12)
  
## Credits

- code to extract the labels and annotations from the .mat files: Devon Yates' code on Kaggle, thanks Devon! https://www.kaggle.com/criticalmassacre/inaccurate-labels-in-stanford-cars-data-set
