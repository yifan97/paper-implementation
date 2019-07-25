This is an implementation of "Masked AutoEncoder for Density Estimation" by Germain et al., 2015. The core idea is that you can turn an auto-encoder into an autoregressive density model just by appropriately masking the connections in the MLP, ordering the input dimensions in some way and making sure that all outputs only depend on inputs earlier in the list. Like other autoregressive models (char-rnn, pixel cnns, etc), evaluating the likelihood is very cheap (a single forward pass), but sampling is linear in the number of dimensions.

The original code is adapted from [Andrej Karpathy](https://github.com/karpathy/pytorch-made)


Some takeways from reading his code:

- activation functions is under torch.nn
- all models should inherent from torch.nn.Module
- register_buffer is usually used to create weigths and bias. The reason to use it can be found [here](https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723)
- if you want to represent weight matrix with information of each layer's unit number, ther is a efficient zip() trick: 

```python
    for h0, h1 in zip(hs, hs[1:])
        h1 = h0
``` 