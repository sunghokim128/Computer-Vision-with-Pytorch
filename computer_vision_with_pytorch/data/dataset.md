This folder is where the dataset would be downloaded.
When it is the fist time executing the code, set `download=True` for the datasets in core.dataset to download the dataset into this folder
<br />
For example: 
```python
torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transf)
```
<br />
The raw file is too big to upload in github repositories and is unnecessary anyway
<br />
<br />
Here are some examples of what these datasets look like:
<br />

## MNIST
![enter image description here](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*9jCey4wywZ4Os7hF.png)

## CIFAR10
![enter image description here](https://production-media.paperswithcode.com/datasets/4fdf2b82-2bc3-4f97-ba51-400322b228b1.png)

## CIFAR100
![enter image description here](https://www.wolframcloud.com/obj/resourcesystem/images/69f/69f1e629-81e6-4eaa-998f-f6734fcd2cb3/492b137097d9b816.png)
