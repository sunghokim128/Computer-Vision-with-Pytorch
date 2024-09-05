# Image Classification with Pytorch

GIF for project demo

GIF![화면 기록 2024-09-05 오후 5 06 09](https://github.com/user-attachments/assets/d58ca2c2-804a-459b-919b-21cf8c5f9a65)

> Training Ending & Graph showing up
> Example of model's prediction and answer


## Project Description
The overall summary of the project can be found in the main.py file.
What I have created here is basically a framework that can be used for training and evaluating pytorch models that are used to sort images.

The datasets I used for this projects are *[MNIST](https://yann.lecun.com/exdb/mnist/), [CIFAR10, CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)*

I started from making a simple FCL (Fully-Connected-Layer) model for a MNIST dataset. Then I tried changing the model architechture by changing the number of nodes in the hidden layers and the depth of the model. Plus I tried using different optimizers (learning rate, momentum), loss functions, activation functions to improve the model's accuracy.

Then I tried mimicking models created by others like AlexNet and ResNet by scaling the parameters down to match the datasets I wanted. ( AlexNet, ResNet uses [224x224] images while images in my datasets are [28x28] or [32x32] )

The log for training results can be found [here](https://www.notion.so/Image-Classification-with-Pytorch-2024-06-de768c6be5174752ba5c240dd3192053)


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
**core**  folder contains functions that are crucial for the framework such as datasets, loss_fuctions and optimizers, plotting, etc

**data**  folder is where all the downloaded datasets are stored

**models** folder contains FCL, AlexNet, ResNet models used to classify images from datasets of MNIST, CIFAR10, CIFAR100

**trained_model_storaged**  folder is where the program stores the trained models (which contains data for milions of parameters)

**util**  folder contains functions that are used but are not crucial such as ones changing time format from [seconds] to [minutes:seconds]

***main***  file summarizes the whole process from deciding the training paraters to showing the train & test results in a plot

***train***  file trains the model by 1 full epoch and returns the average trian & validation loss value for plotting data.
It contains crucial steps in training the model such as loading image data in chuncks with torch.utils.data.Dataloader and computing the cost from loss functions through back-propagation with loss.backward()

***test*** file tests the model with the test dataset and returns the test & validation accuracy value for plotting data.
Top-1, and Top-5 accuracy functions are available.

> Written with [StackEdit](https://stackedit.io/).
