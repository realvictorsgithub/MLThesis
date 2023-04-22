import models
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torch_directml
from engine import train,validate
from dataset import Dataset
from torch.utils.data import DataLoader

#MODEL DEFINITION
matplotlib.style.use('ggplot')

device = torch_directml.device()

model = models.model(pretrained=True, requires_grad=False).to(device)
learningRate = 0.0001
epochs = 20
batch_size=32
optimizer = optim.Adam(model.parameters(), lr = learningRate)
criterion = nn.BCELoss()

#DATA SETUP
train_csv = pd.read_csv("Disease Grading/Groundtruths/IDRiD_Disease Grading_Training Labels.csv")

train_data = Dataset(
    train_csv, train=True, test=False
)

valid_data = Dataset(
    train_csv, train=False, test=False
)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

valid_loader = DataLoader(
    valid_data,
    batch_size=batch_size,
    shuffle=False
)

#RUN SCRIPT
train_loss = []
valid_loss = []

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, train_loader, optimizer, criterion, train_data, device
    )
    valid_epoch_loss = validate(
        model, valid_loader, criterion, valid_data, device
    )

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    print(f"Train loss: {train_epoch_loss:.4f}")
    
    print(f"Valid loss: {valid_epoch_loss:.4f}")

torch.save({
            'epoch':epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,}, "/savedmodel/model.pth")

plt.figure(figsize=(10,7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/savedmodel/loss.png')
plt.show()