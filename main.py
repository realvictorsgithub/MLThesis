
#import all the trash
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import torch_directml
train_csv = pd.read_csv('thesis/Disease Grading/Groundtruths/IDRiD_Disease Grading_Training Labels.csv')
test_csv = pd.read_csv('thesis/Disease Grading/Groundtruths/IDRiD_Disease Grading_Testing Labels.csv')


# ----------------------------------load up the images and visualize graph + distribution of classes
# train_csv = pd.read_csv('thesis/Disease Grading/Groundtruths/IDRiD_Disease Grading_Training Labels.csv')
# test_csv = pd.read_csv('thesis/Disease Grading/Groundtruths/IDRiD_Disease Grading_Testing Labels.csv')

# countTrain = train_csv['Risk of macular edema '].value_counts()
# countTest = test_csv['Risk of macular edema '].value_counts()

# values = ["0","1", "2"]


# for i,x in enumerate(values):
#     countTrain[x] = countTrain.pop(i)
    

# for i,x in enumerate(values):
#     countTest[x] = countTest.pop(i)    

# plt.title("Distribution Training")
# sns.barplot(x =countTrain.index, y=countTrain.values, palette='bright')
# plt.ylabel('number of occurences', fontsize=12)
# plt.xlabel('target classes', fontsize = 12)
# plt.show()



# plt.title("Distribution Testing")
# sns.barplot(x =countTest.index, y=countTest.values, palette='bright')
# plt.ylabel('number of occurences', fontsize=12)
# plt.xlabel('target classes', fontsize = 12)
# plt.show()
# ----------------------------------

#optional visualize data
# ----------------------------------find mean and std of dataset in order to normalize pics

# data_path = "thesis/Disease Grading/Original Images"
# transform_img = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(256),
#     transforms.ToTensor(),
#     # here do not use transforms.Normalize(mean, std)
# ])

# image_data = torchvision.datasets.ImageFolder(
#   root=data_path, transform=transform_img
# )

# image_data_loader = DataLoader(
#     image_data,
#     # batch size is whole dataset
#     batch_size=len(image_data),
#     shuffle=False,
#     num_workers=0)
# def mean_std(loader):
#   images, lebels = next(iter(loader))
#   # shape of images = [b,c,w,h]
#   mean, std = images.mean([0,2,3]), images.std([0,2,3])
#   return mean, std
# mean, std = mean_std(image_data_loader)
# print("mean and std: \n", mean, std)

# ----------------------------------


#dataprocess
   
   
#dataset class
class CreateDataset(Dataset):
    def __init__(self, df_data, data_dir = 'thesis/Disease Grading/Original Images', transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_name,label,label2 = self.df[index]
        img_path = os.path.join(self.data_dir, img_name+'.jpg')
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label, label2
    


#transform
train_transforms = transforms.Compose([
transforms.ToPILImage(),
transforms.Resize((224, 224)),
transforms.RandomHorizontalFlip(p=0.4),
transforms.ToTensor(),
#to normalize, need mean and std from all pics, refer to normalization section
transforms.Normalize(mean=(0.5974, 0.2909, 0.0927), std=(0.2020, 0.1254, 0.0853))
])
    

test_transforms = transforms.Compose([transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize([0.5974, 0.2909, 0.0927],[0.2020, 0.1254, 0.0853])])



train_path = "/home/user/Thesis/thesis/Disease Grading/Original Images/Training Set/"
test_path = "/home/user/Thesis/thesis/Disease Grading/Original Images/Testing Set/"

train_data = CreateDataset(df_data=train_csv, data_dir=train_path, transform=train_transforms)
test_data = CreateDataset(df_data=test_csv, data_dir=test_path, transform=test_transforms)

valid_size = 0.2
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]


train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,sampler=train_sampler)
validloader = torch.utils.data.DataLoader(train_data, batch_size=64, sampler=valid_sampler)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

print(f"training examples contain : {len(train_data)}")
print(f"testing examples contain : {len(test_data)}")

# ---------------------------------- print random sample to verify normalization
#
# images, labels = next(iter(trainloader))
# print(f"Image shape : {images.shape}")
# print(f"Label shape : {labels.shape}")


# # denormalizing images
# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(50)

# grid = torchvision.utils.make_grid(images, nrow = 20, padding = 2)
# plt.figure(figsize = (20, 20))  
# plt.imshow(np.transpose(grid, (1, 2, 0)))   
# print('labels:', labels)    

# class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# images, labels = next(iter(trainloader))
# out = torchvision.utils.make_grid(images)
# imshow(out, title=[class_names[x] for x in labels])

# ----------------------------------

dml = torch_directml.device()

model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

num_features = model.fc.in_features
out_features = 4
out_features2 = 3

model.fc = nn.Sequential(nn.Linear(num_features, 512),nn.ReLU(),nn.Linear(512,out_features),nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr = 0.0001)

scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

model.to(dml)


model_saved = 'classifier.pt'
path = F"/home/user/Thesis/thesis/savedmodel/working{model_saved}"
torch.save(model, path)

def load_model(path):
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  return model

#model = load_model("/home/user/Thesis/thesis/savedmodel/workingclassifier.pt")

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: \n{}".format(pytorch_total_params))


def train_and_test(e):
    epochs = e
    train_losses , test_losses, acc = [] , [], []
    valid_loss_min = np.Inf 
    model.train()
    print("Model Training started.....")
    for epoch in range(epochs):
      running_loss = 0
      batch = 0
      for images , labels, labels2 in trainloader:
        images, labels, labels2 = images.to(dml), labels.to(dml), labels2.to(dml)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch += 1
        if batch % 10 == 0:
            print(f" epoch {epoch + 1} batch {batch} completed") 
      test_loss = 0
      accuracy = 0
      with torch.no_grad():
        print(labels)
        print(f"validation started for {epoch + 1}")
        model.eval() 
        for images , labels, labels2 in validloader:
          images, labels, labels2 = images.to(dml), labels.to(dml), labels2.to(dml)
          logps = model(images) 
          test_loss += criterion(logps,labels) 
          ps = torch.exp(logps)
          top_p , top_class = ps.topk(1,dim=1)
          equals = top_class == labels.view(*top_class.shape)
          accuracy += float(torch.mean(torch.Tensor(equals)))
      train_losses.append(running_loss/len(trainloader))
      test_losses.append(test_loss/len(validloader))
      acc.append(accuracy)
      scheduler.step()
      print("Epoch: {}/{}.. ".format(epoch+1, epochs),"Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),"Valid Loss: {:.3f}.. ".format(test_loss/len(validloader)),
        "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
      model.train() 
      if test_loss/len(validloader) <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,test_loss/len(validloader))) 
        torch.save({
            'epoch': epoch,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_loss_min
            }, path)
        valid_loss_min = test_loss/len(validloader)    
    print('Training Completed Succesfully !')    
    return train_losses, test_losses, acc 

train_losses, valid_losses, acc = train_and_test(5)


plt.plot(train_losses, label='train_')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)


plt.plot(acc, label='accuracy')
plt.legend("")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend(frameon=False)