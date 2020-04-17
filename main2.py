'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

print(111)
import torchvision
import torchvision.transforms as transforms

print(222)
import os
import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

pt = os.path.abspath('.').split('pytorch-cifar')[0]
train_directory = pt + 'dataset/train_data'
valid_directory = pt + 'dataset/test_data'
print(train_directory)
batch_size = 64
num_classes = 52

data = {
    'train': torchvision.datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': torchvision.datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])

}

train_data_size = len(data['train'])
trainloader = torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True)
print(train_data_size)

valid_data_size = len(data['valid'])
testloader = torch.utils.data.DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
print(valid_data_size)

classes = ('1001_1002_1003'
           , '1001_1002_1004'
           , '1001_1002_1005'
           , '1001_1002_1006'
           , '1001_1007_1008'
           , '1001_1007_1009'
           , '1001_1010_1011'
           , '1001_1010_1012'
           , '1013_1014_1016'
           , '1013_1014_1018'
           , '1013_1014_1309'
           , '1013_1014_1311'
           , '1013_1014_1312'
           , '1013_1019_1021'
           , '1013_1307_1313'
           , '1013_1307_1314'
           , '1013_1307_1315'
           , '1013_1307_1316'
           , '1013_1307_1317'
           , '1013_1308_1318'
           , '1013_1308_1319'
           , '1013_1308_1320'
           , '1055_1056_1057'
           , '1055_1056_1058'
           , '1055_1060_1061'
           , '1055_1060_1062'
           , '1055_1060_1063'
           , '1055_1060_1064'
           , '1163_1164_1165'
           , '1163_1164_1166'
           , '1163_1164_1167'
           , '1163_1164_1168'
           , '1163_1164_1169'
           , '1163_1164_1170'
           , '1163_1171_1172'
           , '1163_1171_1173'
           , '1163_1171_1174'
           , '1163_1171_1175'
           , '1163_1171_1176'
           , '1163_1177_1178'
           , '1163_1177_1179'
           , '1163_1177_1180'
           , '1163_1181_1182'
           , '1163_1181_1183'
           , '1163_1181_1184'
           , '1163_1181_1185'
           , '1163_1181_1186'
           , '1163_1187_1188'
           , '1163_1187_1189'
           , '1163_1187_1190'
           , '1163_1187_1191'
           , '1163_1187_1192')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)


# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
