# AdaBound for Keras

Keras port of [AdaBound Optimizer for PyTorch](https://github.com/Luolc/AdaBound), from the paper [Adaptive Gradient Methods with Dynamic Bound of Learning Rate.](https://openreview.net/forum?id=Bkg3g2R9FX)

## Usage

Add the `adabound.py` script to your project, and import it. Can be a dropin replacement for `Adam` Optimizer. 

Also supports `AMSBound` variant of the above, equivalent to `AMSGrad` from Adam.

```python
from adabound import AdaBound

optm = AdaBound(lr=1e-03,
                final_lr=0.1,
                gamma=1e-03,
                weight_decay=0.,
                amsbound=False)
```

## Results

With a wide ResNet 34 and horizontal flips data augmentation, and 100 epochs of training with batchsize 128, it hits 92.16% (called v1).

Weights are available inside the [Releases tab](https://github.com/titu1994/keras-adabound/releases/tag/0.1)

#### NOTE
 - The smaller ResNet 20 models have been removed as they did not perform as expected and were depending on a flaw during the initial implementation. The ResNet 32 shows the actual performance of this optimizer.

> With a small ResNet 20 and width + height data + horizontal flips data augmentation, and 100 epochs of training with batchsize 1024, it hits 89.5% (called v1).

> On a small ResNet 20 with only width and height data augmentations, with batchsize 1024 trained for 100 epochs, the model gets close to 86% on the test set (called v3 below).


### Train Set Accuracy

<img src="https://github.com/titu1994/keras-adabound/blob/master/images/train_acc.PNG?raw=true" height=50% width=100%>

### Train Set Loss

<img src="https://github.com/titu1994/keras-adabound/blob/master/images/train_loss.PNG?raw=true" height=50% width=100%>

### Test Set Accuracy

<img src="https://github.com/titu1994/keras-adabound/blob/master/images/val_acc.PNG?raw=true" height=50% width=100%>

### Test Set Loss

<img src="https://github.com/titu1994/keras-adabound/blob/master/images/val_loss.PNG?raw=true" height=50% width=100%>

# Issue with clipping

Currently dependent on Tensorflow backend for `tf.clip_by_value`. Will be backend independent after next release of Keras.

# Requirements
- Keras 2.2.4+ & Tensorflow 1.12+ (Only supports TF backend for now).
- Numpy
