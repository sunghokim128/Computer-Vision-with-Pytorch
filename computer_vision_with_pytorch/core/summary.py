from torchinfo import summary

from util.dataset_name import dataset_name

def show_summary(args, model):
    '''
    The show_summary function takes in the following parameters and shows the overall structure of the model on the
    console
    :param agrs: parsers from user
    :param model: the pytorch model
    '''

    dataset = dataset_name(args.model)
    batch = args.batch

    # images in the MNIST dataset is in greyscale
    if dataset == "mnist":
        summary(model, input_size=(batch, 1, 28, 28))
    # the pixel dimensions are same for cifar10 and cifar100 (3 x 32 x 32)
    # 3 is for the RGB value
    elif dataset == "cifar10" or dataset == "cifar100":
        summary(model, input_size=(batch, 3, 32, 32))