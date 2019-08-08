# rendergan_pytorch

PyTorch implementation of RenderGAN:
<https://www.frontiersin.org/articles/10.3389/frobt.2018.00066/full>

Note: This is not a 1 to 1 reimplementation of the original keras / theano
code. It uses the same approach as described in the paper, but the networks
architectures and hyperparameters are not identical. Furthermore, several
modern techniques such as self-attention, spectral normalization and
pixelshuffle upsampling are used.