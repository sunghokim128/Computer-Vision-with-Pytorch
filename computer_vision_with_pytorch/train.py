import torch
from torch.utils.data import DataLoader  # DataLoader : BatchSize형태로 만들어줌. Dataset : 튜닝에 사용
from tqdm import tqdm
import os

from core.loss_function import loss_function
from core.optimizer import optimizer
from core.dataset import train_val_test_dataset

from util.dataset_name import dataset_name
from test import *

def train(model, device, model_path, args):
    '''
    train 함수가 하는 일:
    1. 위 정보대로 모델을 훈련시켜서 매 epoch 마다 train_loss, validation_loss를 얻고
       매 epoch 마다 test를 진행하여 test_accuracy, validation_accuracy 를 구함
    2. 전체 에폭만큼 훈련시킨 모델을 모델이름_Nepoch.pth 로 저장
    3. validation_loss가 제일 작았던 모델을 모델이름_best_Nepoch.pth 로 저장
    + tqdm을 사용하여 iteration별 loss 및 전체적인 진행상황을 확인할 수 있음
    + return (train_loss, validation_loss, test_accuracy, validation_accuracy)
    '''

    model.to(device) # moves the model over to MPS
    dataset = dataset_name(args.model)

    epoch = args.epoch
    batch = args.batch
    criterion = loss_function(args.loss_function)
    navigator = optimizer(args.optimizer, model)
    trainset, validationset, testset = train_val_test_dataset(dataset)

    # move train_set, validation_set onto the DataLoader in Batches that's defined above
    trainloader = DataLoader(trainset, batch_size=batch)
    validloader = DataLoader(validationset, batch_size=batch)

    # Four lists are created. Each save the designated value in every epoch to hand over the data to the plot function
    train_loss = []
    validation_loss = []
    test_accuracy = []
    validation_accuracy = []

    bar = tqdm(range(1, epoch + 1)) # tqdm allows to check the training progress from the console
    for Epoch in bar:

        # train_loss calculation & model training
        all_train_loss = 0
        for batch_index, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            # Loss value of last batch is trimmed out for exact calculation of average loss
            if batch_index + 1 < len(trainloader):
                all_train_loss += loss

            navigator.zero_grad()
            loss.backward()
            navigator.step()

            bar.set_description('Training Loss in epoch {}: '.format(Epoch) + str(round(loss.item(), 4)))

        # 해당 epoch의 train_loss 값들의 평균을 저장
        avg_train_loss = all_train_loss / (len(trainloader) - 1)
        train_loss.append(round(avg_train_loss.item(), 4))


        # validation_loss 계산
        all_vali_loss = 0
        for batch_index, (data, target) in enumerate(validloader):
            v_data, v_target = data.to(device), target.to(device)
            v_output = model(v_data)
            v_loss = criterion(v_output, v_target)
            if batch_index + 1 < len(validloader):
                all_vali_loss += v_loss
        # 해당 epoch의 validation_loss 값들의 평균을 저장
        avg_vali_loss = all_vali_loss / (len(validloader) - 1)
        validation_loss.append(round(avg_vali_loss.item(), 4))

        #매 epoch 마다 모델을 accuracy를 측정해서 test_data, validation_data의 accuracy를 각각의 리스트에 추가
        if args.top1:
            test_accuracy_val, validation_accuracy_val = top1_test(model, device, testset, validationset)
        else:
            test_accuracy_val, validation_accuracy_val = top5_test(model, device, testset, validationset)

        test_accuracy.append(test_accuracy_val)
        validation_accuracy.append(validation_accuracy_val)


        # 시스템 정지를 대비해서 매 epoch이 끝날 때 마다 모델 저장
        torch.save(model.state_dict(), model_path + "_{}epoch.pth".format(Epoch))
        if Epoch > 1:
            os.remove(model_path + "_{}epoch.pth".format(Epoch - 1))

    # 안내멘트
    print('Saving training results at :', model_path + "_{}epoch.pth".format(epoch))


    return (train_loss, validation_loss, test_accuracy, validation_accuracy)