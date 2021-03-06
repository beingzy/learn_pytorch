from __future__ import print_function

import argparse
import loguru

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self, *arg, **kwargs):
        super(Net, self).__init__(*arg, **kwargs)
        self._add_nnblocks_to_attrs()

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

    def _add_nnblocks_to_attrs(self):
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)


def train(
    args: dict, 
    model: nn.Module, 
    device: str, 
    train_loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    epoch: int):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) # negative log-likelihood
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            log_msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 
                100. * batch_idx / len(train_loader), loss.item())
            print(log_msg)
            if args.dry_run:
                break


def test(model: nn.Module, device: str, test_loader: torch.utils.data.DataLoader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    log_msg = '\nTests set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 
        100. * correct / len(test_loader.dataset)
    )
    print(log_msg)


def transform_featurizer():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    return transform


def build_argparser():
    parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', 
        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', 
        help='input batch size for testing (default: 1,000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N', 
        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', 
        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', 
        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, 
        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False, 
        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=42, metavar='S', 
        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N', 
        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, 
        help='For saving the current Model')
    return parser


def get_train_test_kwargs_from_args(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {
            'nun_workers': 1, 
            'pin_memory': True, 
            'shuffle': True,
            }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    return train_kwargs, test_kwargs


def gen_model_fname(args):
    """ generate a file name (str) which will be used 
        to store the trained model.
    """
    def float2formated_str(x:float):
        return str('{:.2f}'.format(x)).replace('.', 'd')
    
    return "mnist_cnn_epoch_{epoch}_lr_{lr}.pt".format(
        epoch=args.epochs,
        lr=float2formated_str(args.lr)
    )


def main():
    parser = build_argparser()
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs, test_kwargs = get_train_test_kwargs_from_args(args)

    transform = transform_featurizer()
    dataset1 = datasets.MNIST('../data', train=True, download=True,
        transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, download=True,
        transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        model_fname = gen_model_fname(args)
        torch.save(model.state_dict(), f"./models/{model_fname}")


if __name__ == "__main__":
    main()