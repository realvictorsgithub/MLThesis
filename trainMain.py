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
#device = torch.device('cpu')
#model = models.model(pretrained=True, requires_grad=False).to(device)
#model = models.model(pretrained=True, requires_grad=False).to(device)
model_CNN = models.CNN1(True).to(device)

learningRate = 0.0001
momentum_value = 0.9
epochs = 20
batch_size=32
#optimizer = optim.Adam(model.parameters(), lr = learningRate)
#criterion = nn.BCELoss()
criterion_multioutput = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_CNN.parameters(), lr=learningRate, momentum=momentum_value)



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
        model_CNN, train_loader, optimizer, criterion_multioutput, train_data, device
    )
    valid_epoch_loss = validate(
        model_CNN, valid_loader, criterion_multioutput, valid_data, device
    )

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    print(f"Train loss: {train_epoch_loss:.4f}")
    
    print(f"Valid loss: {valid_epoch_loss:.4f}")

torch.save({
            'epoch':epochs,
            'model_state_dict': model_CNN.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion_multioutput,}, "/home/user/Thesis/MLThesis/savedmodel/model.pth")

plt.figure(figsize=(10,7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/home/user/Thesis/MLThesis/savedmodel/loss.png')
plt.show()