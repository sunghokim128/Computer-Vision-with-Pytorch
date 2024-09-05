import torch.optim as optim

def optimizer(optimizer_name, model):
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.5)