import torch
from utils import load_config
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


config = load_config(config_path="config.yaml")

dataset = datasets.ImageFolder(root=config["root"],
                               transform=transforms.Compose([
                                   transforms.Resize(config["image_size"]),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                        std=(0.5, 0.5, 0.5))
                               ])
                               )

dataloader = DataLoader(dataset, batch_size=config["batch_size"],
                        shuffle=True, num_workers=config["workers"])



