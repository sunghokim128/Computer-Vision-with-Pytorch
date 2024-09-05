import matplotlib.pyplot as plt
from util.show_as_minute import show_as_minutes

def plot(train_loss: list, validation_loss: list, test_accuracy: list, validation_accuracy: list, total_time: int):
    '''
    This function helps visualize the results from training & testing by showing the loss and accuracy by each Epoch
    training_loss & validation_loss is plotted in one graph, and test_accuracy & validation_accuracy in the other.

    :param train_loss: The difference between the model's prediction and actual answer computed by the loss_function
    :param validation_loss: same thing as train_loss but computed from validation dataset
    :param test_accuracy: The ratio of how many cases are predicted correctly
    :param validation_accuracy: same thing as test_accuracy but computed from validation dataset
    :param total_time: The total amount of time taken from beginning of training to plotting
    '''

    epoch = len(train_loss)

    # Set figure (graph) size
    plt.figure(figsize=(14, 6))

    # First graph shows train_loss, validation_loss by epochs
    plt.subplot(1, 2, 1)
    plt.plot([i for i in range(1, epoch + 1)], train_loss, label="train_loss")
    plt.plot([i for i in range(1, epoch + 1)], validation_loss, label="validation_loss")
    plt.legend()
    plt.title('Train & Validation Loss by Epochs')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Plot only if the values for test_accuracy, validation_accuracy is given
    if test_accuracy:
        plt.subplot(1, 2, 2)
        plt.plot([i for i in range(1, epoch + 1)], test_accuracy, label="test_accuracy")
        plt.plot([i for i in range(1, epoch + 1)], validation_accuracy, label="validation_accuracy")
        plt.legend()
        plt.title('Test & Validation Accuracy ')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        # adjust space between two graphs
        plt.subplots_adjust(wspace=0.3)  # wspace로 좌우 간격 조절

    # Show the total time spent in [min:sec] form
    plt.suptitle(show_as_minutes(total_time))

    plt.show()
