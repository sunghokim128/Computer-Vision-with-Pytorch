When it is the first time executing the code, set `download=True` for the datasets in core.dataset
For example: 

	   torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transf)
