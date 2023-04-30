import os
import yaml
import shutil
import torch
from torch import nn
from typing import Dict


class Flatten(nn.Module):
    @staticmethod
    def forward(input: torch.Tensor) -> torch.Tensor:
        """
        :param input: tensor with shape: B x C x H x W
        :return: B x C * H * W
        """
        return torch.flatten(input, start_dim=1)


def weight_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, val=0)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as fh:
        data = yaml.safe_load(fh)
    return data


def rename_files(source_folder: str, destination_folder: str) -> None:
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for idx, filename in enumerate(os.listdir(source_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, str(idx) + ".jpg")
            shutil.copyfile(source_path, destination_path)


def main():
    source_folder = "data/pokemon"
    destination_folder = "data/new_pokemon"
    rename_files(source_folder, destination_folder)


if __name__ == "__main__":
    main()
