import torch.nn as nn

def loss_function(loss_function_name: str):
    '''
    This function takes in the desired loss_function name in string and returns the loss_function
    A loss function calculates the magnitude of how wrong the model's prediction of a certain case is
    A continuous decrease in loss value suggests the model is learning well

    :param loss_function_name: ("mse" / "cross_entropy")
    :return: torch.nn.functional.SOME_LOSS_FUNCTION
    '''
    if loss_function_name == "mse":
        return nn.functional.mse_loss
    elif loss_function_name == "cross_entropy":
        return nn.functional.cross_entropy