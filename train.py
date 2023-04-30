import os
import torch
import datetime
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from utils import load_config
from dataset import ImageDataset
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter


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
    fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=device)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizerG = Adam(params=generator.parameters(),
                      lr=LEARNING_RATE,
                      betas=(BETA1, 0.999))
    optimizerD = Adam(params=discriminator.parameters(),
                      lr=LEARNING_RATE,
                      betas=(BETA1, 0.999))

    generator.train()
    discriminator.train()

    # tensorboard --logdir="runs"
    global_step = 0
    writer = SummaryWriter()

    for idx_e, epoch in tqdm(enumerate(range(EPOCHS)), desc="Training", total=EPOCHS):
        for idx_b, batch in enumerate(dataloader):
            real_images = batch.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1, dtype=torch.float, device=device)
            fake_labels = torch.zeros(batch_size, 1, dtype=torch.float, device=device)

            # train discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # train discriminator with real images
            optimizerD.zero_grad()
            real_outputs = discriminator(real_images)
            lossD_real = criterion(real_outputs, real_labels)
            lossD_real.backward()
            # train discriminator with fake images
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_images = generator(noise)
            fake_outputs = discriminator(fake_images)
            lossD_fake = criterion(fake_outputs, fake_labels)
            lossD_fake.backward()
            optimizerD.step()

            # train generator: maximize log(D(G(z)))
            optimizerG.zero_grad()
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_images = generator(noise)
            fake_outputs = discriminator(fake_images)
            lossG_real = criterion(fake_outputs, real_labels)
            lossG_real.backward()
            optimizerG.step()

            # update tensorboard
            if global_step % 5 == 0:
                lossD = (lossD_real + lossD_fake).detach().cpu().item()
                lossG = lossG_real.detach().cpu().item()

                writer.add_scalar(tag="Discriminator loss: log(D(x)) + log(1 - D(G(z)))",
                                  scalar_value=lossD, global_step=global_step)

                writer.add_scalar(tag="Generator loss: log(D(G(z)))",
                                  scalar_value=lossG, global_step=global_step)

                with torch.no_grad():
                    fake_images = generator(fixed_noise).detach().cpu()
                    image_grid = make_grid(fake_images, nrow=4, normalize=True)
                    writer.add_image(tag="Generated images", img_tensor=image_grid, global_step=global_step)

            global_step += 1

    writer.close()

    path = "trained_models"
    if not os.path.exists(path):
        os.makedirs(path)
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_name = f"{os.path.basename(config['root'])}_{date}.pt"
    model_path = os.path.join(path, model_name)
    torch.save(generator.state_dict(), model_path)


if __name__ == "__main__":
    main()
