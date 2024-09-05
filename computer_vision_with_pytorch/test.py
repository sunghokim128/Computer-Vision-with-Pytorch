import torch
from tqdm import tqdm

def top1_test(model, device, test_dataset, validation_dataset):
    '''
    test_dataset, validation_dataset 을 받아서 현재 모델의 accuracy를 각각 계산해서 float를 가진 튜플 형태로
    return (test_accuracy, validation_accuracy)

    + test 는 batch_size가 1이기 때문에 전체 정확도를 계산할 때 [맞은 총 개수 / 데이터의 총 개수]로 이루어짐. 따라서 배치 끝자락 걱정 ㄴㄴ
    '''

    # 추론 단계에서는 shuffle 수행하지 않음
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model.eval()  # eval 설정해야 dropout, batch_normalization 등을 해제

    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="test accuracy calculation", leave=False)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.max(1, keepdim=True)[1]  # .max를 수행하면 [제일 확률이 높은 값, 인덱스]를 반환함
            if target == prediction:
                correct += 1

    test_accuracy = round(correct / len(test_loader) * 100, 3)
    print("test accuracy: {}".format(test_accuracy), end="")

    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(
                tqdm(validation_loader, desc="validation accuracy calculation", leave=False)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.max(1, keepdim=True)[1]  # .max를 수행하면 [제일 확률이 높은 값, 인덱스]를 반환함
            if target == prediction:
                correct += 1

    validation_accuracy = round(correct / len(validation_dataset) * 100, 3)
    print("validation accuracy: {}\n".format(validation_accuracy))

    return (test_accuracy, validation_accuracy)


def top5_test(model, DEVICE, test_dataset, validation_dataset):
    '''
    test_dataset, validation_dataset 을 받아서 현재 모델의 accuracy를 각각 계산해서 float를 가진 튜플 형태로
    return (test_accuracy, validation_accuracy)
    '''

    # 추론 단계에서는 shuffle 수행하지 않음
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model.eval()  # eval 설정해야 dropout, batch_normalization 등을 해제

    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="test accuracy calculation", leave=False)):
            data, target = data.to(DEVICE), target.to(DEVICE)
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
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            prediction = output.topk(5)[1]
            if target in prediction:
                correct += 1
    validation_accuracy = round(correct / len(validation_dataset) * 100, 3)
    print("validation accuracy: {}\n".format(validation_accuracy))

    return (test_accuracy, validation_accuracy)