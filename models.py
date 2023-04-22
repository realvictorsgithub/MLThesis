from torchvision import models as models

import torch.nn as nn

def model(pretrained, requires_grad):
    model = models.resnet101(progress=True, pretrained = pretrained)
    #to freeze hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad=False
    
    elif requires_grad==True:
        for param in model.parameters():
            param.requires_grad=True

    #amountof classes
    model.fc=nn.Linear(2048,2)
    return model