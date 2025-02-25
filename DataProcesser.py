from matplotlib import image
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import xml.etree.ElementTree as ET

class DataProcess(Dataset):
    def __init__(self,args,root_path,ano_path,Frames,transform = None):
        super(DataProcess,self).__init__()
        self.img_path = self.filelist(root_path, Frames)
        self.ano_path = ano_path
        self.transform = transform
        self.tokensize = args.patch_size
        self.imagesize = args.image_size
        self.tokenRawTotal = self.imagesize // self.tokensize

    def filelist(self, root_path, FramesNumber, file_type=('bmp' or 'img')):
        img_path = []  
        # if (len(root_path.split("/")) == 3):
        if (len(root_path.split("\\")) == 3):
           # trainset
            for dirs in os.listdir(root_path):
                datalist = [files for files in os.listdir(os.path.join(root_path, dirs))]
                datalist.sort(key=lambda x: int(x.split('.')[0]))
                datalist = [os.path.join(root_path,dirs,i) for i in datalist]
                datalist = [datalist[i : i+FramesNumber] for i in range(0,len(datalist)-FramesNumber)]
                for item in datalist:
                    img_path.append(item)

            #validset
        else:
            datalist = [files for files in os.listdir(root_path)]
            datalist.sort(key=lambda x: int(x.split('.')[0]))
            datalist = [os.path.join(root_path,i) for i in datalist]
            datalist = [datalist[i : i+FramesNumber] for i in range(0,len(datalist)-FramesNumber)]
            for item in datalist:
                img_path.append(item)
        return img_path

    def xmllist(self,root_path,file_type=('xml')):
        return [os.path.join(root_path,f) for root,dirs,files in os.walk(root_path) for f in files if f.endswith(file_type)]
        
    def __len__(self):
        # return len(self.img_path)                      
        return 1

    def __getitem__(self, index):
        img_file = self.img_path[index]
        xml_file = os.path.join(self.ano_path,'\\'.join(img_file[0].split('\\')[-2:]).replace('.bmp','.xml')) # Windows
        # xml_file = os.path.join(self.ano_path,'/'.join(img_file[0].split('/')[-2:]).replace('.bmp','.xml')) # Linux
        
        assert os.path.exists(xml_file),"{} 文件没有发现!".format(xml_file)
        split = torch.zeros(self.imagesize, self.imagesize)
        xy_center = self.get_center(xml_file)
        for _ in xy_center:
            split[_[1]][_[0]] = 1
        target = int(xy_center[0][1] * 256 + xy_center[0][0])
        targets = {}
        targets['label'] = target
        targets['img_path'] = img_file
        targets['split'] = split
        # 使用灰度图
        # img = [self.transform(Image.open(item).convert("L")) for item in img_file]
        img = [self.transform(Image.open(item)) for item in img_file]
        # for image in img:
        #     image = self.getMeanStd(image)
        return img, targets
    
    def get_center(self,xml_file):
        assert True
        tree = ET.parse(xml_file)
        root = tree.getroot()
        xy_center_list = []
        for tag in root:
            if tag.tag == "object":
                for b_tag in tag:
                    if b_tag.tag == "centerbox":
                        x_center = int(b_tag.findtext("x"))
                        y_center = int(b_tag.findtext("y"))
                        xy_center_list.append((x_center, y_center))
        return xy_center_list 
    
    # def get_center(self,xml_file):
    #     assert True
    #     tree = ET.parse(xml_file)
    #     root = tree.getroot()
    #     for child in root.iter('centerbox'):
    #         x_center = int(child.find('x').text)
    #         y_center = int(child.find('y').text)
    #     return (x_center,y_center)

    def calPostionIndex(self,location):
        if len(location) == 1:
            x,y = int(location[0][0]),int(location[0][1])
            xPostionIndex = x % self.tokensize 
            yPostionIndex = y % self.tokensize
            xyPosintionIndex = yPostionIndex * self.tokensize + xPostionIndex
            return xyPosintionIndex
        else:
            x1,y1 = int(location[0][0]),int(location[0][1])
            x2,y2 = int(location[1][0]),int(location[1][1])
            xPostionIndex1 = x1 % self.tokensize 
            xPostionIndex2 = x2 % self.tokensize 
            yPostionIndex1 = y1 % self.tokensize
            yPostionIndex2 = y2 % self.tokensize
            xyPosintionIndex1 = yPostionIndex1 * self.tokensize + xPostionIndex1
            xyPosintionIndex2 = yPostionIndex2 * self.tokensize + xPostionIndex2
            return xyPosintionIndex1, xyPosintionIndex2
    
    def calTokenIndex(self,location):
        if len(location) == 1:
            x,y = int(location[0][0]),int(location[0][1])
            x_index = x // self.tokensize 
            y_index = y // self.tokensize 
            tokenIndex = self.tokenRawTotal * y_index + x_index
            return tokenIndex
        else:
            x1, y1 = int(location[0][0]), int(location[0][1])
            x2, y2 = int(location[1][0]), int(location[1][1])
            x1_index = x1 // self.tokensize 
            y1_index = y1 // self.tokensize 
            x2_index = x2 // self.tokensize 
            y2_index = y2 // self.tokensize 
            tokenIndex1 = self.tokenRawTotal * y1_index + x1_index
            tokenIndex2 = self.tokenRawTotal * y2_index + x2_index
            return tokenIndex1, tokenIndex2
       

    def getMeanStd(self,image):
        std = 1 if image.std() == 0 else image.std()
        retImage = (image - image.mean()) / std
        # retImage = torch.zeros_like(image)
        # for cIndex in range(retImage.shape[0]):
        #     for y in range(self.tokensize ):
        #         for x in range(self.tokensize ):
        #             x_begin = x * self.tokensize 
        #             x_end = (x+1) * self.tokensize 
        #             y_begin = y * self.tokensize 
        #             y_end = (y+1)* self.tokensize 
        #             imagetoken = image[cIndex,x_begin:x_end,y_begin:y_end]
        #             std = 1 if imagetoken.std() == 0 else imagetoken.std()

        #             imagetoken = (imagetoken - imagetoken.mean()) / std
        #             retImage[cIndex,x_begin:x_end,y_begin:y_end] = torch.as_tensor(imagetoken)
        return retImage

def buildDataLoader(args,train_path,valid_path,Frames,ano_path,transform):
    dataset = DataProcess(args, train_path, ano_path=ano_path, Frames=Frames, transform=transform)
    valset = DataProcess(args, valid_path, ano_path=ano_path, Frames=Frames, transform=transform)
    trainloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    validloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=8, drop_last=True,pin_memory=True)
    return trainloader,validloader
    
    # valset = DataProcess(valid_path,ano_path=ano_path,transform=transform)
    # train_size = int(len(dataset) * 0.8)  #这里train_size是一个长度矢量，并非是比例，我们将训练和测试进行8/2划分
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False)
    # return train_loader, test_loader
    