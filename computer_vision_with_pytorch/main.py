import time
import torch


# 내가 만든 모듈
from train import train
from core.plot import plot
from core.summary import show_summary
from models.FCL import *

from util.dataset_name import dataset_name

import argparse

if __name__ == '__main__':

    # 프로그램 시작시간 체크
    start_time = time.time()

    # 디바이스 설정
    device = torch.device("mps")

    # argparse 로 에폭, 배치 사이즈, loss function, learning rate, momentum 등 학습에 필요한 요소를 받아오기
    # dataset을 따로 arg로 넘기지 않는 이유는 훈련을 시킬때 dataset과 모델을 매칭하지 않는 human error가 일어나는것을 방지하기 위함
    '''
    cross_entropy는 내부적으로 softmax를 적용한다.
    target은 one-hot 형태가 아니라 class lable index 형태로 입력되어야 하고
    model을 통과한 output은 softmax를 통과시키지 말고 raw data 를 그대로 넣어주어야 함

    MSELoss는 target을 one-hot 인코딩 하고, output을 softmax 적용한 값을 사용해야 하는데
    이 특성때문에 같은 train 모듈에서 MSELoss를 같이 사용할 수 없다

    따라서 여기서는 MSE는 깔끔하게 포기하고 넘어가보도록 하자
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", help="Epoch = One full cycle of the total dataset", type=int)
    parser.add_argument("-b", "--batch", help="Batch = The size you want to chop the epoch into", type=int)
    parser.add_argument("-l", "--loss_function", help="Just use cross_entropy for now", type=str, default="cross_entropy")
    parser.add_argument("-o", "--optimizer", help="Choose from sgd / adam", type=str)
    parser.add_argument("-m", "--model", help="Format: [datasetName_modelArchitecture]", type=str)
    parser.add_argument("-t", "--top1", help="Activate if you want top-1 accuracy", type=str, default=None)
    args = parser.parse_args()

    # 학습에 사용할 모델설정. 데이터셋과 맞는 모델을 사용하고 있는지 확인할 것, 끝에 () 붙이는거 확인
    model = eval(args.model + "()")

    # 모델 이름은 [dataset_model] 형식으로 지음. ".pth" 등은 train 함수에서 알아서 붙여주도록 되어있으니 여기서 .pth를 안 붙여도 됨
    model_name = input("Write Model Name as [dataset_model]: ")

    # trained_model 폴더에 모델을 저장하기 위해 디렉토리를 모델 이름 앞에 붙여줌
    model_path = "/Users/apple/PycharmProjects/computer_vision_with_pytorch/trained_model_storage/" + model_name




    '''
    주요 기능 1: pytorch 라이브러리의 info 모듈에서 summary함수로 모델의 전체적인 모습을 확인 가능
    '''
    show_summary(args, model)

    if args.top1:
        print("\nTest result will show Top-1 accuracy\n")
    else:
        print("\nTest result will show Top-5 accuracy\n")

    '''
    주요 기능 2: 모델을 훈련시키고 그 과정에서 구해지는 데이터셋별 loss, accuracy 값을 리스트 형태로 반환
    + batch 끝자락에서 남는 개수만큼 길이를 재서 loss를 구하기 때문에 그걸로 인한 loss값의 변질은 걱정 ㄴㄴ

    train 함수가 하는 일:
    1. 위 정보대로 모델을 훈련시켜서 매 epoch 마다 train_loss, validation_loss를 얻고
       매 epoch 마다 test를 진행하여 test_accuracy, validation_accuracy 를 구함
    2. 전체 에폭만큼 훈련시킨 모델을 모델이름_Nepoch.pth 로 저장
    3. validation_loss가 제일 작았던 모델을 모델이름_best_Nepoch.pth 로 저장
    + tqdm을 사용하여 iteration별 loss 및 전체적인 진행상황을 확인할 수 있음
    + return (train_loss, validation_loss, test_accuracy_list, validation_accuracy_list)
    '''
    results = train(model, device, model_path, args)


    '''
    주요 기능 3: 결과를 시각화해서 보여줌

    plot 함수가 하는 일:
    epoch별 train_loss, validation_loss 를 받아서 그래프로 나타냄
    epoch별 test_accuracy, validation_accuracy를 받아서 그래프로 나타냄
    '''
    plot(results, args.epoch, time.time() - start_time)