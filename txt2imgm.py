# Tiny Text-to-Image GAN Training Script

import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# CONFIG
IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50
LATENT_DIM = 100
LABEL_DIM = 1  # simulate "text" using one float
DATASET_DIR = "./sample_images"  # your training image folder

# DATASET
class ImageTextDataset(Dataset):
    def __init__(self, image_folder):
        self.images = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor([index % 10 / 10.0], dtype=torch.float32)  # Fake label
        return label, img

    def __len__(self):
        return len(self.images)

# MODELS
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + LABEL_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * IMAGE_SIZE * IMAGE_SIZE),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        return self.model(x).view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE + LABEL_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        x = torch.cat([img.view(img.size(0), -1), labels], dim=1)
        return self.model(x)

# TRAINING
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageTextDataset(DATASET_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(EPOCHS):
        for labels, real_imgs in dataloader:
            batch_size = real_imgs.size(0)
            real_imgs, labels = real_imgs.to(device), labels.to(device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM).to(device)
            gen_imgs = generator(z, labels)
            g_loss = criterion(discriminator(gen_imgs, labels), torch.ones(batch_size, 1).to(device))
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_imgs, labels), torch.ones(batch_size, 1).to(device))
            fake_loss = criterion(discriminator(gen_imgs.detach(), labels), torch.zeros(batch_size, 1).to(device))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        print(f"[Epoch {epoch+1}/{EPOCHS}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(generator.state_dict(), "models/text2img_gen.pt")
    print("Model saved to models/text2img_gen.pt")

# RUN
if __name__ == "__main__":
    train()
