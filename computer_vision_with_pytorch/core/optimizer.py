import torch.optim as optim

def set_optimizer(optimizer_name: str, model):
    '''
    This function takes the desired optimizer name in str and a pytorch model, then returns the desired
    optimizer with the model's parameters taken as the param_group

    Try visualizing the model's learning as someone's journey of finding the lowest point in a valley
    the optimizer is basically the navigator; it decides which way to go and how far
    In more technical terms, the optimizer tweaks the parameters (weights, biases) of the model to the opposite
    direction of the derivative of the loss function with respect to the parameters (in order to reduce the loss value)
    The chain rule enables parameters in every layer to be adjusted this way

    :param optimizer_name: ("sgd")
    :param model: a PyTorch model
    :return: optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    '''
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
