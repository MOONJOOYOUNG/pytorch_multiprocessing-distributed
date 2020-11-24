import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.distributed as dist

def get_loader(args, rank, world_size):
    mean = [0.5, 0.5, 0.5]
    stdv = [0.5, 0.5, 0.5]


    train_transform = transforms.Compose([transforms.Resize((32,32)),
                                          transforms.transforms.RandomCrop(32, padding=8),
                                          transforms.transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=stdv)])

    test_transform = transforms.Compose([transforms.Resize((32,32)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=stdv)])

    train_set = datasets.CIFAR10(root='./cifar10_data',
                                 train=True,
                                 transform=train_transform,
                                 download=False)
    test_set = datasets.CIFAR10(root='./cifar10_data',
                                train=False,
                                transform=test_transform,
                                download=False)
    print(1111111111111)
    print(range, args.world_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                    rank=rank, num_replicas=args.world_size,
                                                                    shuffle=True)

    valid_sampler = torch.utils.data.distributed.DistributedSampler(test_set,
                                                                    rank=rank, num_replicas=args.world_size,
                                                                    shuffle=True)

    batch_size = int(args.batch_size/args.world_size)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=int(batch_size),
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=int(batch_size),
                                               shuffle=False,
                                               num_workers=4,
                                              pin_memory=True,
                                               sampler=valid_sampler)
    if dist.get_rank() == 0:
        print("-------------------Make loader-------------------")
        print('Train Dataset :', len(train_loader.dataset),
              '   Test Dataset :', len(test_loader.dataset))

    return train_loader, test_loader