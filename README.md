# Image Classification with Pytorch

## Project Demo
![start_training](https://github.com/user-attachments/assets/0eaf6880-0078-450c-bb4c-402ef91a8fd7)
^ start training with designated parameters

![end_training_with_plot](https://github.com/user-attachments/assets/67832b10-48b9-4b5d-a3cc-94d15ff6a71d)
^ end training & testing with plot summary



## Project Description
The overall summary of the project can be found in the main.py file.
What I have created here is basically a framework that can be used for training and evaluating pytorch models that are used to sort images.

The datasets I used for this projects are *[MNIST](https://www.tensorflow.org/datasets/catalog/mnist), [CIFAR10, CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)*

<img width="434" alt="스크린샷 2024-09-05 오후 4 33 59" src="https://github.com/user-attachments/assets/3588417d-b4c3-4133-9378-dd8f3d1e0623">
<img width="483" alt="스크린샷 2024-09-05 오후 5 49 51" src="https://github.com/user-attachments/assets/b6e5ee8f-9c07-46b2-8742-d51cf2bd1528">


I started from making a simple FCL (Fully-Connected-Layer) model for a MNIST dataset. Then I tried changing the model architechture by changing the number of nodes in the hidden layers and the depth of the model. Plus I tried using different optimizers (learning rate, momentum), loss functions, activation functions to improve the model's accuracy.

Then I tried mimicking models created by others like AlexNet and ResNet by scaling the parameters down to match the datasets I wanted. ( AlexNet, ResNet uses [224x224] images while images in my datasets are [28x28] or [32x32] )

The log for training results can be found [here](https://www.notion.so/Image-Classification-with-Pytorch-2024-06-2024-08-de768c6be5174752ba5c240dd3192053#ce777ef6b89447e9904bfd1ffddfeff2)


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
**[core](https://github.com/sunghokim128/Computer-Vision-with-Pytorch/tree/main/computer_vision_with_pytorch/core)**  folder contains functions that are crucial for the framework such as datasets, loss_fuctions and optimizers, plotting, etc

**[data](https://github.com/sunghokim128/Computer-Vision-with-Pytorch/tree/main/computer_vision_with_pytorch/data)**  folder is where all the downloaded datasets are stored

**[models](https://github.com/sunghokim128/Computer-Vision-with-Pytorch/tree/main/computer_vision_with_pytorch/models)** folder contains FCL, AlexNet, ResNet models used to classify images from datasets of MNIST, CIFAR10, CIFAR100

**[trained_model_storaged](https://github.com/sunghokim128/Computer-Vision-with-Pytorch/tree/main/computer_vision_with_pytorch/trained_model_storage)**  folder is where the program stores the trained models (which contains data for milions of parameters)

**[util](https://github.com/sunghokim128/Computer-Vision-with-Pytorch/tree/main/computer_vision_with_pytorch/util)**  folder contains functions that are used but are not crucial such as ones changing time format from [seconds] to [minutes:seconds]

***[main](https://github.com/sunghokim128/Computer-Vision-with-Pytorch/blob/main/computer_vision_with_pytorch/main.py)***  file summarizes the whole process from deciding the training paraters to showing the train & test results in a plot

***[test](https://github.com/sunghokim128/Computer-Vision-with-Pytorch/blob/main/computer_vision_with_pytorch/test.py)*** file tests the model with the test dataset and returns the test & validation accuracy value for plotting data.
Top-1, and Top-5 accuracy functions are available.

***[train](https://github.com/sunghokim128/Computer-Vision-with-Pytorch/blob/main/computer_vision_with_pytorch/train.py)***  file trains the model by 1 full epoch and returns the average trian & validation loss value for plotting data.
It contains crucial steps in training the model such as loading image data in chuncks with torch.utils.data.Dataloader and computing the cost from loss functions through back-propagation with loss.backward()

> Written with [StackEdit](https://stackedit.io/).
