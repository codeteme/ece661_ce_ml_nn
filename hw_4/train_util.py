import math
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from FP_layers import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


class CIFAR10Loader:
    """
    CIFAR-10 kept as one tensor on the device, with the augmentation done there batch by batch.
    It gives the same data as the torchvision pipeline used before
    (RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor, Normalize),
    without the per-image work on the CPU that made training slow.
    Use it like a DataLoader: iterate to get (inputs, targets), len() gives the number of batches.
    """
    def __init__(self, train, batch_size, shuffle, augment, device):
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True)
        self.images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous().to(device)  # uint8, N x 3 x 32 x 32
        self.labels = torch.tensor(dataset.targets).to(device)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.device = device

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        n = len(self.labels)
        order = torch.randperm(n) if self.shuffle else torch.arange(n)
        order = order.to(self.device)
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            inputs = self.images[idx].float().div_(255)
            if self.augment:
                inputs = random_crop_flip(inputs)
            inputs = (inputs - self.mean) / self.std
            yield inputs, self.labels[idx]


def random_crop_flip(x):
    """
    RandomCrop(32, padding=4) and RandomHorizontalFlip for a batch of images in [0, 1].
    Each image gets its own crop position and flip, as in torchvision.
    """
    n = x.size(0)
    padded = F.pad(x, (4, 4, 4, 4))  # zero padding, before normalization as in torchvision
    offset_y = torch.randint(0, 9, (n,), device=x.device)
    offset_x = torch.randint(0, 9, (n,), device=x.device)
    steps = torch.arange(32, device=x.device)
    rows = (offset_y[:, None] + steps)[:, None, :, None]
    cols = (offset_x[:, None] + steps)[:, None, None, :]
    batch = torch.arange(n, device=x.device)[:, None, None, None]
    channel = torch.arange(3, device=x.device)[None, :, None, None]
    out = padded[batch, channel, rows, cols]
    flip = torch.rand(n, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


_loaders = {}

def get_loaders(batch_size=256, device=device):
    """
    Training and test loaders for CIFAR-10 on the given device.
    In hw4.ipynb (Lab 2), they can replace the torchvision datasets and DataLoaders:
        trainloader, testloader = get_loaders(batch_size=batch_size, device=device)
    """
    device = torch.device(device)
    key = (batch_size, str(device))
    if key not in _loaders:
        _loaders[key] = (CIFAR10Loader(True, batch_size, shuffle=True, augment=True, device=device),
                         CIFAR10Loader(False, 1000, shuffle=False, augment=False, device=device))
    return _loaders[key]


def evaluate(net, testloader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    return test_loss / num_val_steps, correct / total


def train_epoch(net, trainloader, criterion, optimizer, batch_size, global_steps, log_every_n, weight_mask=None):
    """
    One epoch of training. If weight_mask is given, weights that are zero in the mask are kept as zero.
    Loss and accuracy are summed on the device and only read back when they are printed.
    """
    net.train()
    train_loss = torch.zeros((), device=trainloader.device)
    correct = torch.zeros((), device=trainloader.device)
    total = 0
    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        if weight_mask is not None:
            with torch.no_grad():
                for name, layer in net.named_modules():
                    if name in weight_mask:
                        layer.weight.data *= weight_mask[name]
        train_loss += loss.detach()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum()
        global_steps += 1

        if global_steps % log_every_n == 0:
            end = time.time()
            num_examples_per_second = log_every_n * batch_size / (end - start)
            print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                  % (global_steps, train_loss.item() / (batch_idx + 1), (correct.item() / total), num_examples_per_second))
            start = time.time()
    return global_steps


def train(net, epochs, batch_size, lr, reg, log_every_n=50):
    """
    Training a network
    :param net: Network for training
    :param epochs: Number of epochs in total.
    :param batch_size: Batch size for training.
    """
    print('==> Preparing data..')
    device = next(net.parameters()).device
    trainloader, testloader = get_loaders(batch_size, device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.875, weight_decay=reg, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)

    global_steps = 0

    for epoch in range(start_epoch, epochs):
        print('\nEpoch: %d' % epoch)
        global_steps = train_epoch(net, trainloader, criterion, optimizer, batch_size, global_steps, log_every_n)
        scheduler.step()

        test_loss, val_acc = evaluate(net, testloader, criterion)
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving...")
            torch.save(net.state_dict(), "pretrained_model.pt")


def finetune(net, epochs, batch_size, lr, reg, log_every_n=50, keep_sparsity=False):
    """
    Finetuning a network
    :param keep_sparsity: If True, weights that are zero before finetuning are kept as zero
    throughout finetuning, so a pruned model stays pruned.
    """
    print('==> Preparing data..')
    device = next(net.parameters()).device
    trainloader, testloader = get_loaders(batch_size, device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Record the position of nonzero weights, so pruned weights can be kept as zero
    weight_mask = None
    if keep_sparsity:
        weight_mask = {}
        for name, layer in net.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weight_mask[name] = (layer.weight.data != 0).float()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.875, weight_decay=reg, nesterov=False)

    global_steps = 0

    for epoch in range(start_epoch, epochs):
        print('\nEpoch: %d' % epoch)
        global_steps = train_epoch(net, trainloader, criterion, optimizer, batch_size, global_steps, log_every_n, weight_mask)

        test_loss, val_acc = evaluate(net, testloader, criterion)
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving...")
            torch.save(net.state_dict(), "quantized_net_after_finetune.pt")

def test(net):
    device = next(net.parameters()).device
    _, testloader = get_loaders(device=device)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()

    test_loss, val_acc = evaluate(net, testloader, criterion)
    print("Test Loss=%.4f, Test accuracy=%.4f" % (test_loss, val_acc))
