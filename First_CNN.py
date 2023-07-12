import torch
import torch as T
import torchvision as tv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os #формат изображений HWC!!!!!!!!!!
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

class Data4Classes(T.utils.data.Dataset):
    def __init__(self, path1:str, path2:str, path3:str, path4:str):
        super().__init__()

        self.path1 = path1
        self.path2 = path2
        self.path3 = path3
        self.path4 = path4

        self.list1 = sorted(os.listdir(path1))
        self.list2 = sorted(os.listdir(path2))
        self.list3 = sorted(os.listdir(path3))
        self.list4 = sorted(os.listdir(path4))

    def __len__(self):
        return len(self.list1) + len(self.list2) + len(self.list3) + len(self.list4)

    def __getitem__(self, item_name:str):

        if 'StAureus' in item_name:
            class_id = 0
            img_path = os.path.join(self.path1, item_name)
        if 'StHominis' in item_name:
            class_id = 1
            img_path = os.path.join(self.path2, item_name)
        if 'StPasteuri' in item_name:
            class_id = 2
            img_path = os.path.join(self.path3, item_name)
        if 'StEpid' in item_name:
            class_id = 3
            img_path = os.path.join(self.path4, item_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)
        img = img/255.0

        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
        img = img.transpose((2,0,1))   # переводим в CHW

        t_img = T.from_numpy(img)
        t_class_id = T.tensor(class_id)

        return {'img': t_img,
                'ID': t_class_id}

Aureus_path_train = r'C:\Users\Ivan\PycharmProjects\Microbe_CNN\datasets\StaphCNN\StAureus\StAureus whole\StAureus train'
Hominis_path_train = r'C:\Users\Ivan\PycharmProjects\Microbe_CNN\datasets\StaphCNN\StHominis\StHominis whole\StHominis train'
Pasteuri_path_train = r'C:\Users\Ivan\PycharmProjects\Microbe_CNN\datasets\StaphCNN\StPasteuri\StPasteuri train'
Epidermidis_path_train = r'C:\Users\Ivan\PycharmProjects\Microbe_CNN\datasets\StaphCNN\StEpid\StEpid whole\StEpid train'

Aureus_path_test = r'C:\Users\Ivan\PycharmProjects\Microbe_CNN\datasets\StaphCNN\StAureus\StAureus whole\StAureus test'
Hominis_path_test = r'C:\Users\Ivan\PycharmProjects\Microbe_CNN\datasets\StaphCNN\StHominis\StHominis whole\StHominis test'
Pasteuri_path_test = r'C:\Users\Ivan\PycharmProjects\Microbe_CNN\datasets\StaphCNN\StPasteuri\StPasteuri test'
Epidermidis_path_test = r'C:\Users\Ivan\PycharmProjects\Microbe_CNN\datasets\StaphCNN\StEpid\StEpid whole\StEpid test'

DS_train = Data4Classes(Aureus_path_train, Hominis_path_train, Pasteuri_path_train, Epidermidis_path_train)
DS_test = Data4Classes(Aureus_path_test, Hominis_path_test, Pasteuri_path_test, Epidermidis_path_test)
# pas = DS_train.getimg('StAureus 32 (1).jpg')
# print(pas['img'])

# plt.imshow(pas['img'])
# plt.show()


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.actfunc = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool2d(3,1)
        self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3, 32, 5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(3, 32, 5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(3, 32, 5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(3, 32, 5, stride=1, padding=0)

        self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(64, 10)
        self.linear2 = nn.Linear(10, 4)

    def forward(self, ds):

        out = self.conv1(ds)
        out = self.actfunc(out)
        out = self.pool(out)

        out = self.conv2(ds)
        out = self.actfunc(out)
        out = self.pool(out)

        out = self.conv3(ds)
        out = self.actfunc(out)
        out = self.pool(out)

        out = self.conv4(ds)
        out = self.actfunc(out)
        out = self.pool(out)

        out = self.conv5(ds)
        out = self.actfunc(out)
        out = self.pool(out)

        out = self.adaptivepool(out)
        out = self.flat(out)
        out = self.linear1(out)
        out = self.actfunc(out)
        out = self.linear2(out)

        return out



if __name__ == "__main__":


    batch_size = 16

    train_loader = T.utils.data.DataLoader(
        DS_train,
        shuffle = True,
        batch_size = batch_size,
        num_workers = 1
    )

    test_loader = T.utils.data.DataLoader(
        DS_test,
        shuffle = True,
        batch_size = batch_size,
        num_workers = 1
    )


    Model = ConvNet()

    loss_func = nn.CrossEntropyLoss()
    optimizer = T.optim.Adam(Model.parameters(), lr=0.001, betas=(0.9, 0.999))

    def accuracy(pred, label):
        acc = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
        return acc

    def count_parameters(Model):
        return sum(p.numel() for p in Model.parameters() if p.requires_grad)


    epochs = 5


    for epoch in range(epochs):
        loss_val = 0
        acc_val = 0
        for sample in (pbar := tqdm(test_loader)):
            img, label = sample['img'], sample['ID']

            optimizer.zero_grad()

            label = F.one_hot(label, 4).float()
            pred = Model(img)

            loss = loss_func(pred, label)
            loss.backward()
            loss_item = loss.item()
            loss_val += loss_item

            optimizer.step()

            acc_current = accuracy(pred, label)
            acc_val += acc_current

        pbar.set_discription(f'loss: {loss_item:.3f} \n ',
                             f'accuracy: {acc_current:.3f}')
        print(loss_val/len(train_loader))
        print(acc_val/len(test_loader))
