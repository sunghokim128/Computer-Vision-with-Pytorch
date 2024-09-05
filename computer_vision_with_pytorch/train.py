from torch.utils.data import DataLoader
from tqdm import tqdm


from core.loss_function import loss_function
from core.optimizer import set_optimizer
from core.dataset import train_val_test_dataset

from util.dataset_name import dataset_name

def train(model, device, args):
    """
    This function takes the model, device, args and trains the model according to the parser value

    :param model: pytorch model
    :param device: "mps"
    :param args: contains epoch, batch, loss_function, optimizer, model
    :return: (avg_train_loss, avg_vali_loss)
    """

    model.to(device) # moves the model over to MPS
    dataset = dataset_name(args.model)

    batch = args.batch
    criterion = loss_function(args.loss_function)
    optimizer = set_optimizer(args.optimizer, model)
    trainset, validationset, testset = train_val_test_dataset(dataset)

    # move train_set, validation_set onto the DataLoader in Batches that's defined above
    trainloader = DataLoader(trainset, batch_size=batch)
    validloader = DataLoader(validationset, batch_size=batch)


    # train_loss calculation & model training
    all_train_loss = 0

    bar = tqdm(trainloader, leave=False)
    for batch_index, (data, target) in enumerate(bar):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        # Loss value of last batch is trimmed out for exact calculation of average loss
        if batch_index + 1 < len(trainloader):
            all_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        bar.set_description("Training Loss " + str(round(loss.item(), 4)))

    avg_train_loss = all_train_loss / (len(trainloader) - 1)
    print("avg_train_loss: {}".format(avg_train_loss), end="")


    # validation_loss calculation
    all_vali_loss = 0
    for batch_index, (data, target) in enumerate(tqdm(validloader, leave=False)):
        v_data, v_target = data.to(device), target.to(device)
        v_output = model(v_data)
        v_loss = criterion(v_output, v_target)
        if batch_index + 1 < len(validloader):
            all_vali_loss += v_loss.item()
    avg_vali_loss = all_vali_loss / (len(validloader) - 1)

    return (avg_train_loss, avg_vali_loss)