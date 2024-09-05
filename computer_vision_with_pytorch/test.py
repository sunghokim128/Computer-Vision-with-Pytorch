import torch
from tqdm import tqdm
from core.dataset import dataset

from util.dataset_name import dataset_name

def top5_test(model, device, args):
    """
    This function takes the model, device, args and returns the test_accuracy, validation_accuracy of the model on a
    certain dataset in a form of tuple of lists containing float value.
    Top-5 accuracy means that the answer was within the top 5 choice of the model

    :param model: pytorch model
    :param device: "mps"
    :param args: parser from user
    :return: (test_accuracy, validation_accuracy)
    """

    datasetName = dataset_name(args.model)

    test_dataset = dataset("test", datasetName)
    validation_dataset = dataset("validation", datasetName)

    # No shuffle is needed in evaluation process
    # batch_size=1 indicates one image is passed at a time
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model.eval()  # .eval() deactivates dropout, batch_normalization, etc

    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="test accuracy calculation", leave=False)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.topk(5)[1]
            if target in prediction:
                correct += 1

    test_accuracy = round(correct / len(test_loader) * 100, 3)
    print("test accuracy: {}".format(test_accuracy),end="")

    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(
                tqdm(validation_loader, desc="validation accuracy calculation", leave=False)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.topk(5)[1]
            if target in prediction:
                correct += 1
    validation_accuracy = round(correct / len(validation_dataset) * 100, 3)
    print("validation accuracy: {}\n".format(validation_accuracy))

    return (test_accuracy, validation_accuracy)



def top1_test(model, DEVICE, args):
    """
    This function takes the model, device, args and returns the test_accuracy, validation_accuracy of the model on a
    certain dataset in a form of tuple of lists containing float value.
    Top-1 accuracy means that the model's top choice was the answer

    :param model: pytorch model
    :param device: "mps"
    :param args: parser from user
    :return: (test_accuracy, validation_accuracy)
    """

    datasetName = dataset_name(args.model)

    test_dataset = dataset("test", datasetName)
    validation_dataset = dataset("validation", datasetName)

    # No shuffle is needed in evaluation process
    # batch_size=1 indicates one image is passed at a time
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model.eval()  # .eval() deactivates dropout, batch_normalization, etc

    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="test accuracy calculation", leave=False)):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            prediction = output.max(1, keepdim=True)[1]
            if target in prediction:
                correct += 1

    test_accuracy = round(correct / len(test_loader) * 100, 3)
    print("test accuracy: {}".format(test_accuracy),end="")

    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(
                tqdm(validation_loader, desc="validation accuracy calculation", leave=False)):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            prediction = output.max(1, keepdim=True)[1]
            if target in prediction:
                correct += 1
    validation_accuracy = round(correct / len(validation_dataset) * 100, 3)
    print("validation accuracy: {}\n".format(validation_accuracy))

    return (test_accuracy, validation_accuracy)