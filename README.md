# Image Classification with Pytorch

## Project Description
The overall summary of the project can be found in the main.py file.
What I have created here is basically a framework that can be used for training and evaluating pytorch models that are used to sort images.

The datasets I used for this projects are *[MNIST](https://yann.lecun.com/exdb/mnist/), [CIFAR10, CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)*

I started from making a simple FCL (Fully-Connected-Layer) model for a MNIST dataset. Then I tried changing the model architechture by changing the number of nodes in the hidden layers and the depth of the model. Plus I tried using different optimizers (learning rate, momentum), loss functions, activation functions to improve the model's accuracy.

Then I tried mimicking models created by others like AlexNet and ResNet by scaling the parameters down to match the datasets I wanted. (AlexNet, ResNet uses 224x224 images while images in my datasets are 28x28 or 32x32)

The log for training results can be found [here](https://www.notion.so/Computer-Vision-with-Pytorch-2024-06-de768c6be5174752ba5c240dd3192053)


## Requirements
```
PyTorch
Numpy
Torchvision
Torchinfo
Matplotlib
Tqdm
```

## Code Description
**core**  folder contains modules that are crucial for the framework such as datasets, loss_fuctions and optimizers, plotting, etc

**data**  folder is where all the downloaded datasets are stored

**trained_model_storaged**  folder is where the framework stores the trained models information (hundreds and thousands of weight and bias values)

**util**  folder contains functions that are used but are not crucial such as changing time format from [seconds] to [hours:minutes:seconds]

**train, test**  are the files needed for training, testing, and additionally training a certain model which are all used in the main file.

The whole process of deciding the model, training parameters, and then training and testing it can be found in the  **main**  file



> Written with [StackEdit](https://stackedit.io/).
