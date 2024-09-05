When it is the first time executing the code, set `download=True` for the datasets in core.dataset
For example: 
```python
torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transf)
```

The raw file is too big to upload in github repositories and is unnecessary anyway

Here are some examples of what these datasets look like:

## MNIST
![enter image description here](https://storage.googleapis.com/tfds-data/visualization/fig/mnist-3.0.1.png)

## CIFAR10
![enter image description here](https://production-media.paperswithcode.com/datasets/4fdf2b82-2bc3-4f97-ba51-400322b228b1.png)

## CIFAR100
![enter image description here](https://www.wolframcloud.com/obj/resourcesystem/images/69f/69f1e629-81e6-4eaa-998f-f6734fcd2cb3/492b137097d9b816.png)
