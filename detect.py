import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse

from DataProcesser import buildDataLoader
from torchvision import transforms
from tqdm import tqdm
from model import CovNet
# from loss import SoftIoULoss
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from utils import calPosition, is_pixel_in_center

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--train_path' , type = str, default = '.\\dataset\\trainset', help="训练集路径")
    parse.add_argument('--valid_path', type = str, default='.\\dataset\\validset_final')
    parse.add_argument('--ano_path', type=str, default='.\\dataset\\Annotations' )
    parse.add_argument('--model_stage',  type = str, default = 'valid',help="train valid PDFA")
    parse.add_argument('--epochs',  type = int,  default = 501)
    parse.add_argument('--image_size' , type = int, default = 256)
    parse.add_argument('--patch_size', type = int, default = 16)
    parse.add_argument('--Frames', type = int, default = 3)
    parse.add_argument('--Interval_period', type = int, default = 50, help="模型保存间隔周期")
    parse.add_argument('--lr', type=float, default = 1e-4)
    parse.add_argument('--model',  type = str,  default = 'None')
    parse.add_argument('--therehold',  type = float,  default = 13)
    parse.add_argument('--model_value_list', type=list, default=[300])
    args = parse.parse_args(args=[])

    device = 'cuda' if torch.cuda.is_available else 'cpu'
    model = CovNet().to(device)
    transform = transforms.Compose([transforms.ToTensor()])
    training_loader,validation_loader = buildDataLoader(args,train_path=args.train_path, valid_path = args.valid_path, Frames = args.Frames, ano_path = args.ano_path, transform = transform)
    with torch.no_grad():
        model_list = args.model_value_list
        for data in model_list:
            print("开始验证".format(data))
            model.eval()      # Don't need to track gradents for validation
            checkpoint = torch.load("experiment\model_patch_pixel_0.87_0.96.pth".format(data), map_location="cuda")
            model.load_state_dict(checkpoint['m'])
            count_top1 = 0
            count_top5 = 0
            for j, vdata in tqdm(validation_loader):
                vinputs, vlabels = j, vdata
                img0, img1 = vinputs[0].cuda(), vinputs[1].cuda()
                res, top_5 = model(img0, img1)
                top5_pred = {"x_pred":[], "y_pred":[], "pred":[]}
                x_label, y_label = calPosition(int(vlabels['label']), args.image_size)
                top5 = torch.topk(res.flatten(0), 5, largest=True, sorted=True)
                for item in top_5[1]:
                    x_pred, y_pred = calPosition(int(item), args.image_size)
                    top5_pred["x_pred"].append(x_pred)
                    top5_pred["y_pred"].append(y_pred)
                    top5_pred["pred"].append((x_pred, y_pred))
                for i in range(len(top5_pred["pred"])):
                        if is_pixel_in_center(top5_pred["x_pred"][0], top5_pred["y_pred"][0], x_label, y_label, 1 ):
                            count_top1 += 1
                        if is_pixel_in_center(top5_pred["x_pred"][i], top5_pred["y_pred"][i], x_label, y_label, 1 ):
                            count_top5 += 1
                            break
                    
                savefile = vlabels['img_path'][0][-1]
                dirname,pngname = savefile.split('\\')[-2:]
                # dirname,pngname = savefile.split('/')[-2:] # Linux
                dirpath = "anno_img/{}".format(dirname)
                os.makedirs(dirpath,exist_ok=True)
                src_img  = cv2.imread(savefile)
                # top1
                cv2.circle(src_img, (x_label, y_label) ,2,(0,0,255),-1)  
                cv2.circle(src_img, top5_pred["pred"][0] ,2,(0,255,0),-1)  
                # top5
                # for _ in top5_pred["pred"]:
                #     cv2.circle(src_img, _ ,2,(0,255,0),-1)  
                cv2.imwrite('{}/{}'.format(dirpath,pngname),src_img)
            top1_acc = count_top1 / len(validation_loader)
            top5_acc = count_top5 / len(validation_loader)
            with open("record.csv", mode="a") as f:
                f.write("{},{:.3f},{:.3f}\n".format(args.valid_path.split("\\")[-1], top1_acc,top5_acc))      
            print("{}:top1精度:{}".format(args.valid_path.split("\\")[-1], top1_acc))         
            print("{}:top5精度:{}".format(args.valid_path.split("\\")[-1], top5_acc))         
    print('Finished value')

if __name__ == "__main__":
    main()