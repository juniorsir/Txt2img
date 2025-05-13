# Install dependencies
!pip install torch torchvision matplotlib

import torch, torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os, random

# Dummy dataset using labels as text
class FlowerDataset(Dataset):
    def __init__(self, image_folder):
        self.images = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.images[idx]).convert("RGB"))
        label = torch.randint(0, 10, (1,)).float()  # fake text-to-vector
        return label, img

    def __len__(self):
        return len(self.images)

# Generator & Discriminator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(101, 256),
            nn.ReLU(),
            nn.Linear(256, 3*64*64),
            nn.Tanh()
        )

    def forward(self, z, label):
        x = torch.cat([z, label], dim=1)
        return self.fc(x).view(-1, 3, 64, 64)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3*64*64 + 1, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, label):
        x = torch.cat([img.view(img.size(0), -1), label], dim=1)
        return self.model(x)

# Training Loop (simple)
# Save model: torch.save(generator.state_dict(), "text2img_gen.pt")
