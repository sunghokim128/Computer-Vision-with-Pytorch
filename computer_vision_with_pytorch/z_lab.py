from torch.utils.data import DataLoader
from core.dataset import dataset

# Loss_function Demo
# from core.loss_function import loss_function
#
# criterion1 = loss_function("cross_entropy")
# criterion2 = loss_function("mse")
#
# a = torch.tensor([1., 1.])
# b = torch.tensor([0.9, 0.9])
#
# print(criterion1(a,b)) # tensor(1.2477)
# print(criterion2(a,b)) # tensor(0.0100)

mnist_test = dataset("test", "mnist")

train_loader = DataLoader(mnist_test, batch_size=512)
print(len(mnist_test))
print(len(train_loader))