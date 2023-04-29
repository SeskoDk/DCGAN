import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from utils import load_config
from dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from models import Generator, Discriminator


def main():
    config = load_config("config.yaml")

    # Hyperparameters
    BATCH_SIZE = config["batch_size"]
    LATENT_DIM = config["nz"]
    LEARNING_RATE = config["lr"]
    BETA1 = config["beta1"]
    EPOCHS = config["num_epochs"]
    IMAGE_SIZE = config["image_size"]
    WORKERS = config["workers"]

    dataset = ImageDataset(root=config["root"],
                           transform=transforms.Compose([
                               transforms.Resize(IMAGE_SIZE),
                               transforms.CenterCrop(IMAGE_SIZE),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()

    # B x 100 x 1 x 1
    fixed_noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1, device=device)

    # models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # optimizer
    optimizerG = Adam(params=generator.parameters(),
                      lr=LEARNING_RATE,
                      betas=(BETA1, 0.999))
    optimizerD = Adam(params=discriminator.parameters(),
                      lr=LEARNING_RATE,
                      betas=(BETA1, 0.999))

    real = 1
    fake = 0
    real_labels = torch.ones(BATCH_SIZE, real, dtype=torch.float, device=device)
    fake_labels = torch.zeros(BATCH_SIZE, fake, dtype=torch.float, device=device)

    G_losses = []
    D_losses = []
    for idx_e, epoch in tqdm(enumerate(range(EPOCHS))):
        for idx_b, batch in enumerate(dataloader):
            batch = batch.to(device)

            # train discriminator: maximize log(D(x)) + log(1 - D(G(z)))

            # train discriminator with real batch
            optimizerD.zero_grad()

            outputD = discriminator(batch)
            outputD = torch.flatten(outputD, 1)
            lossD_D = criterion(outputD, real_labels)
            lossD_D.backward()
            # train discriminator with fake batch
            noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1, device=device)
            outputG = generator(noise)
            outputD = discriminator(outputG)
            outputD = torch.flatten(outputD, 1)
            lossD_G = criterion(outputD, fake_labels)
            lossD_G.backward()

            #TODO track the loss values
            optimizerD.zero_grad()




if __name__ == "__main__":
    main()

