"""
  train.py

  This module handles the training of the Siamese Neural Network for skin lesion classification.
  Includes dataset loading, model initialization, training loop, and model checkpointing

  Author: Wu Chun Yueh
  Date: 31st October 2025
"""
import torch
from torch import nn, optim
from dataset import SiameseDataset
from model import SiameseNetwork, ContrastiveLoss
from torch.utils.data import DataLoader
from torchvision import transforms
import config

"""
  Device configuration
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

"""
  Hyperparameters
"""
batch_size = 256 # Sutatble for RTX 5090
learning_rate = 1e-4 
num_epochs = 12
margin = 0.5
embedding_dim = 128

"""
  Data transforms normalized all the images
"""
base_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

"""
  Augmentation transforms for minority class melanoma
  These augmentations create variations to prevent overfitting
"""
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

"""
  Create training dataset and dataloader with class balancing
"""
print("\n== Creating Balanced Training Dataset ==")
train_dataset = SiameseDataset(
    image_dir=config.train_image,
    labels_csv=config.train_labels,
    transform=base_transform,
    augment_transform=augment_transform,
    balance_classes = True  # Enable class balancing
)

train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 16,
    pin_memory = True if torch.cuda.is_available() else False,
    prefetch_factor=4
)

print(f'\nTraining dataset size: {len(train_dataset)}')

"""
  Initialize the Siamese Network, loss function, and optimizer
"""
model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
criterion = ContrastiveLoss(margin=margin)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

"""
  Execute a training loop for the Siamese Network
  For each epoch:
    - Set model to training mode and oop over batches from the training DataLoader
    - Move images and labels to the device (GPU if available)
    - Perform a forward pass through the Siamese Network to get embeddings
    - Compute contrastive loss between paired embeddings.
    - Backpropagate the loss and update network weights 
    - Track running loss and print batch wise progress
After each epoch:
    - Compute and print the average loss across all batches.
"""
print("\n=== Starting Training ===")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (img1, img2, label) in enumerate(train_loader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        # Forward pass
        optimizer.zero_grad()
        output1, output2 = model(img1, img2)

        # Calculate loss
        loss = criterion(output1, output2, label)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}] Batch [{i + 1}/{len(train_loader)}] Loss: {loss.item():.4f}')

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'siamese_model_balanced.pth')
print('\n=== Training Complete ===')
print('Model saved as siamese_model_balanced.pth')
