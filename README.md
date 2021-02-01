# pytorch-cifar-resnet18
A trained Pytorch ResNet-18 model to be used in the article ---

# Model Metrics:
- Trained on Cifar-10 Dataset
- Test accuracy of ~ 92%

# How to load model?

```python
import torchvision
import torch

model = torchvision.models.resnet18()
checkpoint = torch.load("/content/checkpoint/ckpt.pth")

model = torch.nn.DataParallel(model)
model.load_state_dict(checkpoint["net"])

model = model.module
```
