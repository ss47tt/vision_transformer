import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm  # Import tqdm for progress bars

def load_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels for ViT
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def modify_vit_for_mnist():
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.hidden_dim, 10)  # Modify output layer for 10 classes
    return model

def train_model(model, train_loader, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Update progress bar with current loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    return model

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct, total = 0, 0
    progress_bar = tqdm(test_loader, desc="Evaluating", leave=True)

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

def main():
    train_loader, test_loader = load_mnist_data()
    model = modify_vit_for_mnist()
    model = train_model(model, train_loader)
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()