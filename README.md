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

With a small ResNet 20 and width + height data + horizontal flips data augmentation, and 100 epochs of training with batchsize 1024, it hits 89.5% (called v1).

On a small ResNet 20 with only width and height data augmentations, with batchsize 1024 trained for 100 epochs, the model gets close to 86% on the test set (called v3 below).


### Test Set Accuracy

<img src="https://github.com/titu1994/keras-adabound/blob/master/images/val_acc.PNG?raw=true" height=50% width=100%>

### Test Set Loss

<img src="https://github.com/titu1994/keras-adabound/blob/master/images/val_loss.PNG?raw=true" height=50% width=100%>

# Issue with clipping

For performance reasons, this currently works only on Tensorflow backend. Keras offers a `K.clip` function in its generic backend, but for some reason requires that the clipping values be only python floating point numbers, whereas Tensorflow can accept Tensors without an issue.

The values for these _can_ be read using `K.eval`, however this incurs a drastic performance cost, especially for the Optimizer which is called every batch for updates in a loop for every variable.

# Requirements
- Keras 2.2.4+ & Tensorflow 1.12+ (Only supports TF backend for now).
- Numpy
