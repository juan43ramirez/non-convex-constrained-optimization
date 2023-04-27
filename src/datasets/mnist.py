import torch
import torch.utils.data as torch_data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def mnist(train_batch_size: int = 100, val_batch_size: int = 100):
    data_path = "../../data"

    transf = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    transform_data = transforms.Compose(transf)

    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform_data)
    train_loader = torch_data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    val_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform_data)
    val_loader = torch_data.DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, 10, (28, 28, 1)
