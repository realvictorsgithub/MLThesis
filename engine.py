import torch
from tqdm import tqdm


def train(model, dataloader, optimizer, criterion, train_data, device):
    print('training')
    model.train()
    counter = 0
    train_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/ dataloader.batch_size)):
        counter += 1
        data, target = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(data)

        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, target)
        train_loss += loss.item()

        loss.backward()

        optimizer.step()

    train_loss_total = train_loss / counter
    return train_loss_total

def validate(model, dataloader, criterion, val_data, device):
    print("validating")
    model.eval()
    counter = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/ dataloader.batch_size)):
            counter+= 1 
            data, target = data['image'].to(device), data['label'].to(device)
            outputs = model(data)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, target)
            val_running_loss+= loss.item()
    
        val_loss = val_running_loss / counter
        return val_loss