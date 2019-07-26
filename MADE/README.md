This is an implementation of "Masked AutoEncoder for Density Estimation" by Germain et al., 2015. The core idea is that you can turn an auto-encoder into an autoregressive density model just by appropriately masking the connections in the MLP, ordering the input dimensions in some way and making sure that all outputs only depend on inputs earlier in the list. Like other autoregressive models (char-rnn, pixel cnns, etc), evaluating the likelihood is very cheap (a single forward pass), but sampling is linear in the number of dimensions.

The original code is adapted from [Andrej Karpathy](https://github.com/karpathy/pytorch-made)


Some **takeways** from reading his code:

- activation functions is under ```torch.nn```
- all models should inherent from ```torch.nn.Module```
- register_buffer is usually used to create weigths and bias. The reason to use it can be found [here](https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723)
- if you want to represent weight matrix with information of each layer's unit number, ther is a efficient zip() trick: 

```python
    for h0, h1 in zip(hs, hs[1:])
        masklinear = MaskedLinear(h0, h1)
``` 

where hs is a list of number of units in each layer, h0 is the number of units in previous layer, h1 is the number of units in next layer

- nn.Sequential takes positional argument *arg or a disctionary
- trick of *None*: adds new axis to original vector
```python
self.m[l-1][:,None]
self.m[l][None,:]
```

- zip() is super useful in many cases, think about how you can leverage its power
- there are two main ways to convert a numpy array to tensor:
```python
tensor1 = torch.from_numpy(arr)
tensor2 = torch.Tensor(arr)
```
- To get the length of each dimension, just use 
```python
tensor.size()  # to get length of all dimensions. 
tensor.size(1) # to get the length of second dimension
```
- logits are unnormalized scores of your model. You apply softmax to it to get a probability distribution
```python
input = torch.randn((3, 2), requires_grad=True)
target = torch.rand((3, 2), requires_grad=False)

# see the difference between two cross entropy
loss1 = F.binary_cross_entropy(F.sigmoid(input), target)
loss2 = F.binary_cross_entropy_with_logits(input, target)

loss1.backward()
loss2.backward()

```