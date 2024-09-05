import torchvision
import torchvision.transforms as tr
from torch.utils.data import random_split

def dataset(dataset_type, dataset_name):
    '''
    dataset:
    it downloads the required dataset (MNIST, CIFAR10, CIFAR100 ... )
    then it transforms the data into Tensors and normalizes them
    it returns the desired dataset with the desired type.
    For example, if you wanted a training dataset for a model using images from CIFAR10
    you would simply have to type dataset(train, cifar10)
    '''

    if dataset_name == "mnist":

        # transform setting
        transf = tr.Compose([tr.ToTensor(),
                             tr.Normalize((0.1307,), (0.3081,))])

        # take 5% from the trainset to create a validation set
        # set download=True if it is first time running the code
        dataSet = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transf)
        train_set, validation_set = random_split(dataSet, [int(len(dataSet) * 0.95), int(len(dataSet) * 0.05)])

        if dataset_type == 'train':
            return train_set
        if dataset_type == 'validation':
            return validation_set
        if dataset_type == 'test':
            return torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transf)


    if dataset_name == "cifar10":

        # transform setting
        transf = tr.Compose([tr.ToTensor(),
                             tr.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

        # take 5% from the trainset to create a validation set
        dataSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transf)
        train_set, validation_set = random_split(dataSet, [int(len(dataSet) * 0.95), int(len(dataSet) * 0.05)])

        if dataset_type == 'train':
            return train_set
        if dataset_type == 'validation':
            return validation_set
        if dataset_type == 'test':
            return torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transf)


    if dataset_name == "cifar100":

        # transform setting
        transf = tr.Compose([tr.ToTensor(),
                             tr.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        # take 5% from the trainset to create a validation set
        dataSet = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transf)
        train_set, validation_set = random_split(dataSet, [int(len(dataSet) * 0.95), int(len(dataSet) * 0.05)])

        if dataset_type == 'train':
            return train_set
        if dataset_type == 'validation':
            return validation_set
        if dataset_type == 'test':
            return torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transf)

'''
In case you need to hand everything over at once
'''
def train_val_test_dataset(dataset_name):
    train_set = dataset("train", dataset_name)
    validation_set = dataset("validation", dataset_name)
    test_set = dataset("test", dataset_name)
    return (train_set, validation_set, test_set)
