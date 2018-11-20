# Deformable Convolutional Networks in PyTorch
This repo is an implementation of [Deformable Convolution](https://arxiv.org/abs/1703.06211).
Ported from author's MXNet [implementation](https://github.com/msracver/Deformable-ConvNets).

# Build

```
sh make.sh
CC=g++ python build.py
```

See `test.py` for example usage.

### Notice
Only `torch.cuda.FloatTensor` is supported.
