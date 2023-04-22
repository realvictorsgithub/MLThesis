import torch
import cv2
import numpy as np 
import torchvision.transforms as transforms

from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, csv, train, test):
        self.csv = csv
        self.test = test
        self.train = train

        self.images_names = self.csv[:]['Image name']
        
        self.labels =  np.array( self.csv.drop(['Image name'], axis=1))
        self.train_ratio = int(0.85 * len(self.csv))
        self.valid_ratio = len(self.csv) - self.train_ratio
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((400, 400)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
        ])
    
    # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {self.images_names.count}")
            self.image_names = list(self.images_names)
            self.labels = list(self.labels)
            # define the training transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])

        elif self.train == False and self.test == False:
            print(f"Number of validation images: {self.valid_ratio}")
            self.image_names = list(self.images_names)
            self.labels = list(self.labels)
            # define the validation transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.ToTensor(),
            ])


        elif self.test == True and self.train == False:
                self.image_names = list(self.image_names[-10:])
                self.labels = list(self.labels[-10:])
                # define the test transforms
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ])

        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self,index):
         image = cv2.imread(f"/home/user/Thesis/MLThesis/Disease Grading/Original Images/Training Set/{self.image_names[index]}.jpg")
         image = self.transform(image)
         targets = self.labels[index]
         return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
         }