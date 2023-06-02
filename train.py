import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2
import AlexNet

class MyDataloader(Dataset):
    def __init__(self,root_dir):
        self.root_dir=root_dir

        # 对数据进行预处理操作，将读取到的PIL格式的图像转换为Tensor，并进行标准化处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 获取数据集中的所有图片名称，以及每张图片所属的类别标签（0表示猫，1表示狗）
        self.img_names = []
        self.labels = []
        for idx, category in enumerate(['daisy', 'dandelion','rose','sunflower','tulip']):
            category_path = os.path.join(self.root_dir, category)
            for img_name in os.listdir(category_path):
                self.img_names.append(os.path.join(category_path, img_name))
                label_vec=[0,0,0,0,0]
                label_vec[idx]=1
                self.labels.append(label_vec)
    def __len__(self):
        return len(self.labels)
    # 根据索引获取数据
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.img_names[idx]
        img = Image.open(img_path).convert('RGB')
        # 将图像进行预处理
        img_tensor = self.transform(img)
        # 获取标签，并将其转换为Tensor格式
        label = self.labels[idx]
        label_tensor = torch.tensor(label)
        return img_tensor, label_tensor

    def test_showitem(self,idx):
        # 测试用于输出数据集中单个图片以及对应的标签并显示
        img_path = os.path.join(self.root_dir, self.img_names[idx])
        mat=cv2.imread(img_path,1)
        print("mat:",mat)
        cv2.imshow("test_showitem",mat)
        label = self.img_names[idx][:-4]
        # print("label:",label)
        # print(self.__getitem__(idx)[0].shape)
        print(self.__getitem__(idx)[1])
        cv2.waitKey(0)



def detect(img_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model_path = 'D:/pythonItem/savedmodel/Flwoer.pth'
    Mymodel = AlexNet.AlexNet()
    if os.path.exists(model_path):
        print("开始加载权重")
        checkpoint = torch.load(model_path)
        Mymodel.load_state_dict(checkpoint['model_state_dict'])
        print("加载完成")
    else:
        print("权重不存在")
        return 0
    img = Image.open(img_path).convert('RGB')
    img_tensor=transform(img)
    img_tensor=img_tensor.unsqueeze(dim=0)
    # print(img_tensor.shape)
    # exit(1)
    out=Mymodel(img_tensor)
    out=nn.functional.softmax(out)
    print("out:",out)


def train(epoch_nums):
    batch=64
    train_dataset = MyDataloader(r"D:\pythonItem\flowerClassification\flowers")
    # train_dataset.test_showitem(100)
    # exit(3)
    train_data_loader = DataLoader(train_dataset, batch_size=batch, num_workers=0, shuffle=True, drop_last=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('当前设备是:',device)
    Mymodel = AlexNet.AlexNet()
    Mymodel.to(device)

    criterion = nn.MultiLabelSoftMarginLoss()  # 损失函数
    optimizer = torch.optim.Adam(Mymodel.parameters(), lr=0.001) # 优化器
    model_path = 'D:/pythonItem/savedmodel/Flwoer.pth'
    if os.path.exists(model_path):
        print('开始加载模型')
        checkpoint = torch.load(model_path)
        Mymodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("模型不存在，开始训练")
    i = 1
    for epoch in range(epoch_nums):
        print("epoch:", epoch)
        running_loss = 0.0
        Mymodel.train()  # 神经网络开启训练模式
        for data in train_data_loader:
            inputs, labels = data
            # print(inputs.shape, labels.shape)
            # exit(2)
            inputs, labels = inputs.to(device), labels.to(device)  # 数据发送到指定设备
            # 每次迭代都要把梯度置零
            optimizer.zero_grad()
            # 前向传播
            outputs = Mymodel(inputs).view(batch, 1, 5)
            # 计算误差
            loss = criterion(outputs, labels)
            # 后向传播
            loss.backward()
            # 优化参数
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 0:
                print("当前loss:", running_loss / 200)
                # 保存模型
                # torch.save(Mymodel.state_dict(), model_path)
                torch.save({
                    'model_state_dict': Mymodel.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, model_path)
                # if (running_loss / 200 <= 0.005):
                #     print("提前结束训练")
                #     exit(2)
                running_loss = 0
            i += 1
        # 每5个epoch 更新学习率
        # if epoch % 5 == 4:
        #     for p in optimizer.param_groups:
        #         p['lr'] *= 0.9

# trainData=MyDataloader(r"D:\pythonItem\flowerClassification\flowers")
# trainData.test_showitem(3800)

train(1000)
# detect(r"D:\pythonItem\flowerClassification\flowers\daisy\2573240560_ff7ffdd449.jpg")