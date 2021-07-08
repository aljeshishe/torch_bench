from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 640*3, 3, 1)
        self.conv2 = nn.Conv2d(640*3, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Dataset:

    def __init__(self, shape, batch_size, count, device):
        self.shape = shape
        self.batch_size = batch_size
        self.count = count
        self.device = device

    def __len__(self):
        return self.count

    def __iter__(self):
        return self._iter()

    def _iter(self):
        for i in range(self.count):
            img_tensor = torch.rand(size=[self.batch_size] + list(self.shape), device=self.device)
            targets_tensor = torch.randint(low=0, high=10, size=(self.batch_size,), device=self.device)
            yield img_tensor, targets_tensor


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST benchmark for cpu/gpu')
    parser.add_argument('-d', '--duration', type=int, default=30, metavar='N',
                        help='benchmark duration')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.set_num_threads(12)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Benchmarking using: {device}')
    if use_cuda:
        print(f'Device: {torch.cuda.get_device_name(0)}')

    train_loader = Dataset(shape=(1, 28, 28), batch_size=args.batch_size, count=10000, device=device)
    model = Net()
    model.to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    start = time.time()
    model.train()

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), mininterval=2):
        if start + args.duration < time.time():
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    passed = time.time() - start
    print(f'{passed:.3f}secs passed. {batch_idx} done. loss: {loss.item():.3f}\n')
    results = {'i7-8550U 12thr(baseline)': 0.512,
               'Tesla T4': 15.673,
               'NVIDIA GeForce RTX 2080 Ti': 25.149,
               'TITAN RTX': 28.133,
               'A100': 114.114,
               'A100 2 DP': 142.715,
               'A100 DDP each': 107.919,
               '4g.20Gb': 62.460,
               '2g.10Gb': 31.220,
               '1g.5Gb': 15.319,
               'current': batch_idx / passed}
    from tabulate import tabulate
    baseline = next(iter(results.values()))
    table = [[name, value, value / baseline * 100] for name, value in results.items()]
    headers = ['name', 'it/s', '% to baseline']
    print(tabulate(table, headers, floatfmt='.3f'))


if __name__ == '__main__':
    main()
