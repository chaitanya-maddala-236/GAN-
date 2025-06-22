import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
learning_rate = 0.0002
num_epochs = 50
noise_dim = 100
img_size = 28
num_classes = 10

# Generator Network
class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        
        # Embedding for class labels
        self.label_emb = nn.Embedding(num_classes, noise_dim)
        
        # Generator layers
        self.model = nn.Sequential(
            nn.Linear(noise_dim * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Embed labels
        label_emb = self.label_emb(labels)
        # Concatenate noise and label embedding
        gen_input = torch.cat([noise, label_emb], dim=1)
        # Generate image
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, img_size, img_size)
        return img

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        
        # Embedding for class labels
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        
        # Discriminator layers
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size * 2, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        # Embed labels
        label_emb = self.label_emb(labels)
        # Concatenate image and label embedding
        disc_input = torch.cat([img_flat, label_emb], dim=1)
        # Discriminate
        validity = self.model(disc_input)
        return validity

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True
)

# Initialize models
generator = Generator(noise_dim, num_classes).to(device)
discriminator = Discriminator(num_classes).to(device)

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Training function
def train_gan():
    generator.train()
    discriminator.train()
    
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            batch_size_current = imgs.size(0)
            
            # Adversarial ground truths
            valid = torch.ones(batch_size_current, 1, requires_grad=False).to(device)
            fake = torch.zeros(batch_size_current, 1, requires_grad=False).to(device)
            
            # Real images and labels
            real_imgs = imgs.to(device)
            real_labels = labels.to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Loss on real images
            real_validity = discriminator(real_imgs, real_labels)
            d_real_loss = adversarial_loss(real_validity, valid)
            
            # Generate fake images
            noise = torch.randn(batch_size_current, noise_dim).to(device)
            fake_labels = torch.randint(0, num_classes, (batch_size_current,)).to(device)
            fake_imgs = generator(noise, fake_labels)
            
            # Loss on fake images
            fake_validity = discriminator(fake_imgs.detach(), fake_labels)
            d_fake_loss = adversarial_loss(fake_validity, fake)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate fake images
            fake_imgs = generator(noise, fake_labels)
            fake_validity = discriminator(fake_imgs, fake_labels)
            g_loss = adversarial_loss(fake_validity, valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # Print progress
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        
        # Save sample images every 10 epochs
        if epoch % 10 == 0:
            save_sample_images(epoch)
    
    # Save trained models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
    print("Training completed and models saved!")

# Function to save sample images
def save_sample_images(epoch):
    generator.eval()
    with torch.no_grad():
        # Generate one sample for each digit
        noise = torch.randn(10, noise_dim).to(device)
        labels = torch.arange(0, 10).to(device)
        fake_imgs = generator(noise, labels)
        
        # Denormalize images
        fake_imgs = fake_imgs * 0.5 + 0.5
        
        # Create grid
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(10):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(fake_imgs[i].cpu().squeeze(), cmap='gray')
            axes[row, col].set_title(f'Digit {i}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'samples_epoch_{epoch}.png')
        plt.close()
    generator.train()

# Function to generate specific digits
def generate_digit(digit, num_samples=5):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim).to(device)
        labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
        fake_imgs = generator(noise, labels)
        
        # Denormalize images
        fake_imgs = fake_imgs * 0.5 + 0.5
        fake_imgs = torch.clamp(fake_imgs, 0, 1)
        
        return fake_imgs.cpu().numpy()

if __name__ == "__main__":
    # Create directories
    os.makedirs('data', exist_ok=True)
    
    print("Starting GAN training...")
    train_gan()
    
    # Test generation
    print("Testing digit generation...")
    for digit in range(10):
        samples = generate_digit(digit, 5)
        print(f"Generated 5 samples for digit {digit}")
