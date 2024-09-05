import torch.nn as nn

def loss_function(loss_function_name):
    if loss_function_name == "mse":
        return nn.functional.mse_loss
    if loss_function_name == "cross_entropy":
        return nn.functional.cross_entropy