import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import os

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the Model (copied from notebook)
class ConvAE(nn.Module):
    def __init__(self, img_size=28, embedding_dim=3):
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        self.losses = []
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), 
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                       
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                       
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Flatten(),                             
            nn.Linear(64 * 7 * 7, embedding_dim)                
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64 * 7 * 7),               
            nn.ReLU(True),
            nn.Unflatten(1, (64, 7, 7)),              
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), 
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0), 
            nn.ReLU(True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1), 
            nn.Sigmoid()                              
        )

    def encode(self, x):
        x = self.encoder(x)
        return x 
    
    def decode(self, x):
        x = self.decoder(x)
        return x 
  
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x 

def main():
    print("Setting up data...")
    img_size = 28
    # Load training data (Subset as in notebook)
    ds_full_train = torchvision.datasets.MNIST('./files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize((img_size,img_size)),
                                torchvision.transforms.ToTensor(),
                                ]))
    ds_train_subset = torch.utils.data.Subset(ds_full_train, range(1024))
    train_dl = torch.utils.data.DataLoader(ds_train_subset, batch_size=16, shuffle=True)
    
    # Load Test Data
    ds_test = torchvision.datasets.MNIST('./files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize((img_size,img_size)),
                                torchvision.transforms.ToTensor(),
                                ]))
    test_dl = torch.utils.data.DataLoader(ds_test, batch_size=256, shuffle=False)
    
    # Initialize Model
    print("Initializing model...")
    net = ConvAE(img_size=img_size, embedding_dim=3)
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Train Model
    print("Training model (100 epochs)...")
    net.train()
    # Using 100 epochs as per notebook
    for epoch in range(100):
        total_loss = 0.
        for x, y in train_dl:
            xhat = net(x)
            l = loss_fn(xhat, x)
            optim.zero_grad()
            for p in net.parameters():
                l += 0.0005*p.norm(2)
            total_loss += l.item()
            l.backward()
            optim.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {total_loss:.4f}")

    # Evaluation: Reconstruction Loss on Test Set
    print("\nEvaluating Reconstruction Loss on Test Set...")
    net.eval()
    test_loss = 0.
    total_samples = 0
    with torch.no_grad():
        for x, y in test_dl:
            xhat = net(x)
            # Use sum or mean? Notebook used BCELoss which defaults to mean.
            # But we want total loss or average per sample? 
            # Standard metric is usually Mean MSE or Mean BCE per pixel or per image.
            # BCELoss is per element usually combined with mean.
            l = loss_fn(xhat, x) 
            test_loss += l.item() * x.size(0)
            total_samples += x.size(0)
    
    avg_test_loss = test_loss / total_samples
    print(f"Test Set Reconstruction Loss (BCE): {avg_test_loss:.6f}")

    # Evaluation: Classification Accuracy via KNN on Embeddings
    print("\nEvaluating Latent Space Classification Accuracy (1-NN)...")
    
    # extracting training embeddings
    train_loader = torch.utils.data.DataLoader(ds_train_subset, batch_size=256, shuffle=False)
    train_embeddings = []
    train_labels = []
    
    with torch.no_grad():
        for x, y in train_loader:
            h = net.encode(x)
            train_embeddings.append(h)
            train_labels.append(y)
            
    train_embeddings = torch.cat(train_embeddings)
    train_labels = torch.cat(train_labels)
    
    # extracted test embeddings
    test_embeddings = []
    test_labels = []
    with torch.no_grad():
        for x, y in test_dl:
            h = net.encode(x)
            test_embeddings.append(h)
            test_labels.append(y)
    
    test_embeddings = torch.cat(test_embeddings)
    test_labels = torch.cat(test_labels)
    
    # 1-NN implementation in Torch
    # Distance matrix: (N_test, N_train)
    # To avoid OOM, do it in batches or loops
    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")
    
    correct = 0
    total = 0
    
    # Process test set in chunks
    chunk_size = 100
    num_test = test_embeddings.size(0)
    
    for i in range(0, num_test, chunk_size):
        end = min(i + chunk_size, num_test)
        batch_emb = test_embeddings[i:end]
        batch_labels = test_labels[i:end]
        
        # batch_emb: (B, D), train_embeddings: (N, D)
        # cdist computes p-norm distance
        dists = torch.cdist(batch_emb, train_embeddings, p=2) 
        
        # Get nearest neighbor indices
        # min over dim=1 (training samples)
        _, indices = torch.min(dists, dim=1)
        
        # Create predicted labels
        pred_labels = train_labels[indices]
        
        correct += (pred_labels == batch_labels).sum().item()
        total += batch_labels.size(0)
        
    accuracy = correct / total * 100
    print(f"1-NN Classification Accuracy using latent embeddings: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
