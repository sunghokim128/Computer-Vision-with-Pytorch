import time
import os
import argparse

# import my functions from other directories
from test import *
from train import train
from core.plot import plot
from util.summary import show_summary
from util.operation import operation

# import my models from other directories
from models.FCL import *
from models.AlexNet import *
from models.ResNet import *

if __name__ == '__main__':

    # Start recording training & testing time
    start_time = time.time()

    # Set device as MPS (for M1 Mac users, Metal Performance Shaders is used for GPU acceleration)
    device = torch.device("mps")

    # By using argparse, users can pass values required for training and testing the model
    # Since the dataset being used must always match the model using it, a separate parser for dataset doesn't exist
    # Instead, the program takes the model name and decides which dataset to use from it
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", help="Epoch = One full cycle of the total dataset", type=int)
    parser.add_argument("-b", "--batch", help="Batch = The size you want to chop the epoch into", type=int)
    parser.add_argument("-l", "--loss_function", help="Just use cross_entropy for now", type=str, default="cross_entropy")
    parser.add_argument("-o", "--optimizer", help="Choose from sgd / adam", type=str)
    parser.add_argument("-m", "--model", help="Format: [datasetName_modelArchitecture]", type=str)
    parser.add_argument("-t", "--top1", help="Activate if you want top-1 accuracy", type=str, default=None)
    parser.add_argument("-O", "--operation", help="1 = Train & Test, 2 = Train Only, 3 = Continued Train", type=int, default=1)
    args = parser.parse_args()

    # declare model
    model = eval(args.model + "()")

    # set the correct directory to save the model file in [trained_model_storage]
    model_path = "/Users/apple/PycharmProjects/computer_vision_with_pytorch/trained_model_storage/" + args.model

    # gives True/False values for the flags below.
    LOAD_PREV_MODEL, DO_TEST = operation(args.operation)

    '''
    Main Function 1: Notification
    the show_summary function shows the overall architecture of the model in the console
    general information about the training & evaluating process is also printed
    plus, the program shows whether it is using top-1 or top-5 accuracy when evaluating the model
    top-5 is the default, and it means the answer was within the top 5 choice of the model
    '''
    show_summary(args, model)
    if LOAD_PREV_MODEL:
        print("\nDO_CONTINUED_TRIANING: {}".format(LOAD_PREV_MODEL))
    if DO_TEST:
        if args.top1:
            print("Test result will show Top-1 accuracy\n")
        else:
            print("Test result will show Top-5 accuracy\n")
    else:
        print("Training Only\n")


    '''
    Main Function 2: Continued Training
    Sometimes, the initial epoch might not provide enough training. 
    In such cases, the Continued training section allows additional training of the designated model.
    '''
    if LOAD_PREV_MODEL:
        file_directory = "/Users/apple/PycharmProjects/computer_vision_with_pytorch/trained_model_storage/"
        file_name = input("model for continued training: ") + ".pth"
        model.load_state_dict(torch.load(file_directory + file_name, weights_only=True))
        new_path = file_directory + file_name.strip("epoch.pth")


    '''
    Main Function 3: Training & Testing (Evaluating)
    in each epoch, the model is trained and returns the average training_loss & validation_loss value
    if DO_TEST=True, the test_accuracy & validation_accuracy values are also returned
    Those values get stored in the four lists declared below and are later passed onto the plot function
    '''
    train_loss = []
    validation_loss = []
    test_accuracy = []
    validation_accuracy = []

    # tqdm visualizes the training & testing process
    bar = tqdm(range(1, args.epoch + 1), desc="Total Epochs")
    for epoch in bar:

        t_loss, val_loss = train(model, device, args)
        train_loss.append(t_loss)
        validation_loss.append(val_loss)

        if DO_TEST:

            if args.top1:
                t_acc, val_acc = top1_test(model, device, args)
                test_accuracy.append(t_acc)
                validation_accuracy.append(val_acc)
            else:
                t_acc, val_acc = top5_test(model, device, args)
                test_accuracy.append(t_acc)
                validation_accuracy.append(val_acc)

        # Even if program shuts down mid-training, the most recent version will be saved in trained_model_storage
        if LOAD_PREV_MODEL:
            torch.save(model.state_dict(), new_path + "+{}epoch.pth".format(epoch))
            if epoch > 1:
                os.remove(new_path + "+{}epoch.pth".format(epoch - 1))
        else:
            torch.save(model.state_dict(), model_path + "_{}epoch.pth".format(epoch))
            if epoch > 1:
                os.remove(model_path + "_{}epoch.pth".format(epoch - 1))

    # Training completion notification
    if LOAD_PREV_MODEL:
        print('Saving training results at :', new_path + "+{}epoch.pth".format(args.epoch))
    else:
        print('Saving training results at :', model_path + "_{}epoch.pth".format(epoch))


    '''
    Main Function 4: Plot
    The train_loss & validation_loss is plotted in one graph, and test_accuracy & validation_accuracy in the other.
    The visualized losses and accuracies helps understanding the result and makes it easier to figure out the problem
    if the outcome wasn't good as expected. For example, if the validation accuracy is going up but the test_accuracy 
    bends down, it might be a problem of overfitting
    Plus, the performance of the model should be determined by validation accuracy since, in real life,  
    the test_dataset would only be available for evaluation after the model training is over
    '''
    plot(train_loss, validation_loss, test_accuracy, validation_accuracy, time.time() - start_time)
