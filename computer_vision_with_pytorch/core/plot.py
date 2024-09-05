import matplotlib.pyplot as plt
from util.show_as_minute import show_as_minutes

def plot(results, epoch, total_time):
    '''
    The plot function helps visualize the results from training & testing by showing the loss and accuracy by each Epoch
    :param results: get values of train_loss_list, validation_loss_list, test_accuracy_list, validation_accuracy_list
                    in a format of tuple of lists
    :param epoch: number of epochs used for training
    :param total_time: total time taken
    :return: Shows two graphs:  One Containing the Train & Validation Loss by each Epoch
                                and the other containing the Test & Validation Accuracy by each Epoch
    '''

    train_loss_list, validation_loss_list, test_accuracy_list, validation_accuracy_list = results

    # 전체 figure 크기 설정 (가로, 세로 크기)
    plt.figure(figsize=(14, 6))

    # 첫 번째 플롯: Train & Validation Loss by Epochs
    plt.subplot(1, 2, 1)
    plt.plot([i for i in range(1, epoch + 1)],train_loss_list, label="train_loss")
    plt.plot([i for i in range(1, epoch + 1)], validation_loss_list, label="validation_loss")
    plt.legend()
    plt.title('Train & Validation Loss by Epochs')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")


    plt.subplot(1, 2, 2)
    plt.plot([i for i in range(1, epoch + 1)], test_accuracy_list, label="test_accuracy")
    plt.plot([i for i in range(1, epoch + 1)], validation_accuracy_list, label="validation_accuracy")
    plt.legend()
    plt.title('Test & Validation Accuracy ')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    # 플롯 간의 간격 조절
    plt.subplots_adjust(wspace=0.3)  # wspace로 좌우 간격 조절

    # 훈련의 총 경과 시간 설정
    plt.suptitle(show_as_minutes(total_time))

    # 그래프를 하나의 창에서 동시에 표시
    plt.show()
