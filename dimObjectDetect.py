import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
from numpy import *
from DataProcesser import buildDataLoader
from torchvision import transforms
from tqdm import tqdm
from model import CovarNet
from loss import SoftIoULoss
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


parse = argparse.ArgumentParser()
parse.add_argument('--train_path' , type = str, default = './trainset', help="训练集路径")
parse.add_argument('--valid_path', type = str, default='./validset')
parse.add_argument('--ano_path', type=str, default='./Annotations' )
parse.add_argument('--model_stage',  type = str, default = 'train',help="train valid PDFA")
parse.add_argument('--epoch',  type = int,  default = 501)
parse.add_argument('--image_size' , type = int, default = 256)
parse.add_argument('--patch_size', type = int, default = 16)
parse.add_argument('--Frames', type = int, default = 2)
parse.add_argument('--Interval_period', type = int, default = 5, help="模型保存间隔周期")
parse.add_argument('--lr', type=float, default = 1e-5)
parse.add_argument('--model',  type = str,  default = 'None')
parse.add_argument('--therehold',  type = float,  default = 13)
parse.add_argument('--model_value_list', type=list, default=[500])
args = parse.parse_args(args=[])

device = 'cuda' if torch.cuda.is_available else 'cpu'
TIMEMARK = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
writer = SummaryWriter("runs/dimOD_{}".format(TIMEMARK))
model = CovarNet().to(device)
transform = transforms.Compose([transforms.ToTensor()])
training_loader,validation_loader = buildDataLoader(args,train_path=args.train_path, valid_path = args.valid_path,                                                                                                                Frames = args.Frames, ano_path = args.ano_path, transform = transform)
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
model_stage = args.model_stage
def is_pixel_in_center(x, y, x_label, y_label, tolerance):
    if x - tolerance <= x_label and x_label <= x + tolerance and y_label - tolerance <= y and y <= y_label + tolerance:
        return True
    else:
        return False

def calPosition(x):
    x_pred = x % 256
    y_pred = x // 256
    return x_pred, y_pred

if model_stage == 'train':
    model.train()
    for epoch in tqdm(range(args.epoch)):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        count = 0
        for i,data in enumerate(training_loader, 0):
            total += 1
            inputs, labels = data
            img0, img1 = inputs[0].cuda(), inputs[1].cuda()
            # label = [labels[i].to(device) for i in labels if i == "label"][0]
            outputs = model(img0, img1)
            loss = loss_fn(outputs, labels['label'].to(device))
            running_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("loss = ",loss.item())
            writer.add_scalar("loss_step", loss, epoch*len(training_loader) + i) 
            writer.flush()  
            if epoch % args.Interval_period == 0:
                dictmodel = {"m": model.state_dict()}
                torch.save(dictmodel,"model_data_{}.pth".format(epoch))
        
        writer.add_scalar("loss_echo", running_loss/total, epoch)
        writer.flush()
    print('Finished Training')

elif model_stage == "valid":
    '''
    value:真实坐标在[256,256]中的值
    true_ciirdinate_index:真实坐标排序后的位置
    '''
    #model_50.pth
    with torch.no_grad():
        model_list = args.model_value_list
        # for epochs in tqdm(range(11)):
        for epochs in model_list:
            print("开始第{}个模型的验证".format(epochs))
            model.eval()      # Don't need to track gradents for validation
            checkpoint = torch.load("model_data_500.pth",map_location="cuda")
            # checkpoint = torch.load("best_model_frame3_num_head8.pth".format(epochs), map_location="cuda")
            model.load_state_dict(checkpoint['m'])
            count_top1 = 0
            count_top5 = 0
            atten_list = []
            for j, vdata in tqdm(validation_loader):
                vinputs, vlabels = j, vdata
                img0, img1 = vinputs[0].to(device), vinputs[1].to(device)
                res = model(img0, img1)
                top1 = int(torch.argmax(res))
                x_pred, y_pred = calPosition(top1)
                x_label, y_label = calPosition(int(vlabels["label"]))
                if is_pixel_in_center(x_pred, y_pred, x_label, y_label, 1):
                    count_top1 += 1
                savefile = vlabels['img_path'][0][-1]
                dirname,pngname = savefile.split('/')[-2:]
                dirpath = "anno_img/{}".format(dirname)
                os.makedirs(dirpath,exist_ok=True)
                src_img  = cv2.imread(savefile)
                cv2.circle(src_img,(x_pred,y_pred),2,(0,255,0),-1)  
                cv2.imwrite('{}/{}'.format(dirpath,pngname),src_img)
            top1_acc = count_top1 / len(validation_loader)
            with open("record.txt", mode="a") as f:
                f.write("{}:top1精度:{}\n".format(args.valid_path.split("/")[-1],top1_acc))
            print("{}:top1精度:{}".format(args.valid_path.split("/")[-1],top1_acc))         
    print('Finished value')

elif model_stage == "PDFA":
    # drawHotPic(model, validation_loader, device, 256)
    with torch.no_grad():
        # model_list = args.model_value_list
        PD_store = []
        FA_store = []
        FA_o_store = []
        for therehold_value in range(0,26):
        # for epochs in model_list:
            # print("开始第{}个模型的验证".format(epochs))
            model.eval()      # Don't need to track gradents for validation
            # checkpoint = torch.load("model_data_{}.pth".format(epochs),map_location="cuda")
            checkpoint = torch.load("best_model_frame1_num_head8.pth", map_location="cuda")
            model.load_state_dict(checkpoint['m'])
            thereold = therehold_value
            print(therehold_value)
            PD_list = []
            FA_list = []
            FA_o_list = []
            for j, vdata in tqdm(validation_loader):
                vinputs, vlabels = j, vdata
                target_num = vlabels['center'].shape[1]
                count = 0
                res,attn,top5 = model(vinputs.cuda())
                attnT = list(torch.where(attn > thereold)[1])
                for i in list(vlabels['center'].squeeze(dim = 0)):
                    for x in attnT:
                        token_Index, templateIndex, x_pred, y_pred  = calPosition(args, int(x))
                        if is_Target(args, 0 ,x_pred, int(i[0]), y_pred, int(i[1])):
                            count += 1
                            break    
                PD = count / target_num                
                FA = (len(attnT) - count) / (args.image_size ** 2)
                if len(attnT) == 0:
                    FA_o = 0
                else:
                    FA_o = (len(attnT) - count) / len(attnT)   
                PD_list.append(PD)
                FA_list.append(FA)
                FA_o_list.append(FA_o)
            with open("PD_FA.txt", mode='a') as f:
                f.write("序列:{}\t阈值:{}\t PD:{}\t FA:{} \t FA_o:{}\n".format(args.valid_path.split("/")[-1], thereold, mean(PD_list), mean(FA_list), mean(FA_o_list)))
            PD_store.append(mean(PD_list))
            FA_store.append(mean(FA_list))
            FA_o_store.append(mean(FA_o_list))
            print(mean(PD_list))
            print(mean(FA_list))
            print(mean(FA_o_list))
            print("Finish")
        with open("Total PD_FA.txt", mode='a') as f:
            f.write(f"FA={FA_store}\n")
            f.write(f"PD={PD_store}\n")
            f.write(f"FA_o={FA_o_store}\n")
        

                   