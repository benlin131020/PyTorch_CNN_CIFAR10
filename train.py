import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

BATCH_SIZE = 128
LEARNING_RATE = 1e-2  # 学习率
EPOCHS = 10
torch.manual_seed(1)  # 设置随机数种子，确保结果可重复

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):  # 28x28x1
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),  # 28 x28
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 14 x 14
            nn.Conv2d(6, 16, 5, stride=1, padding=0),  # 10 * 10*16
            nn.ReLU(True), nn.MaxPool2d(2, 2))  # 5x5x16

        self.fc = nn.Sequential(
            nn.Linear(576, 120),  # 400 = 5 * 5 * 16
            nn.Linear(120, 84),
            nn.Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)  # 400 = 5 * 5 * 16, 
        out = self.fc(out)
        return out

model = Cnn(3, 10).to(device)# 图片大小是28x28, 10
# 打印模型
print(model)
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# 开始训练
for epoch in range(EPOCHS):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):  # 批处理
        img, label = data
        img = Variable(img).to(device)
        label = Variable(label).to(device)
        # 前向传播 
        out = model(img)
        loss = criterion(out, label)  # loss
        running_loss += loss.data.item() * label.size(0)  # total loss , 由于loss 是batch 取均值的，需要把batch size 乘回去
        _, pred = torch.max(out, 1)  # 预测结果
        num_correct = (pred == label).sum()  # 正确结果的num
        # accuracy = (pred == label).float().mean() #正确率
        running_acc += num_correct.data.item()  # 正确结果的总数
        # 后向传播
        optimizer.zero_grad()  # 梯度清零，以免影响其他batch
        loss.backward()  # 后向传播，计算梯度
        optimizer.step()  # 梯度更新

        # if i % 300 == 0:
        #    print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
        #        epoch + 1, num_epoches, running_loss / (batch_size * i),
        #        running_acc / (batch_size * i)))
    # 打印一个循环后，训练集合上的loss 和 正确率
    print('Train Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))

# 模型测试， 由于训练和测试 BatchNorm, Dropout配置不同，需要说明是否模型测试
model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:  # test set 批处理
    img, label = data
    img = Variable(img, volatile=True).to(device)  # volatile 确定你是否不调用.backward(), 测试中不需要
    label = Variable(label, volatile=True).to(device)
    out = model(img)  # 前向算法 
    loss = criterion(out, label)  # 计算 loss
    eval_loss += loss.data.item() * label.size(0)  # total loss
    _, pred = torch.max(out, 1)  # 预测结果
    num_correct = (pred == label).sum()  # 正确结果
    eval_acc += num_correct.data.item()  # 正确结果总数

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_dataset)), eval_acc * 1.0 / (len(test_dataset))))

# 保存模型
torch.save(model.state_dict(), './cnn.pth')
