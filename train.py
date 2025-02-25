import os
import torch
import torch.nn as nn
import numpy as np
import argparse

from DataProcesser import buildDataLoader
from torchvision import transforms
from tqdm import tqdm
from model import CovNet
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from utils import calPosition, is_pixel_in_center

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--train_path' , type = str, default = '.\\dataset\\trainset', help="训练集路径")
    parse.add_argument('--valid_path', type = str, default='.\\dataset\\validset_final')
    parse.add_argument('--ano_path', type=str, default='.\\dataset\\Annotations' )
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
    TIMEMARK = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer = SummaryWriter("runs/dimOD_{}".format(TIMEMARK))
    model = CovNet().to(device)
    transform = transforms.Compose([transforms.ToTensor()])
    training_loader,validation_loader = buildDataLoader(args,train_path=args.train_path, valid_path = args.valid_path, Frames = args.Frames, ano_path = args.ano_path, transform = transform)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    model.train()
    for epoch in tqdm(range(args.epochs)):  # loop over the dataset multiple times
        running_loss = 0.0
        total = 0
        for i,data in enumerate(training_loader, 0):
            total += 1
            inputs, labels = data
            img0, img1 = inputs[0].cuda(), inputs[1].cuda()
            outputs = model(img0, img1)
            loss = loss_fn(outputs[0], labels['label'].to(device))
            running_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch={epoch}/{args.epochs} batch={i}/{len(training_loader)} loss = {loss.item()}", end="\r")
            writer.add_scalar("loss_step", loss, epoch*len(training_loader) + i) 
            writer.flush()
            if epoch % args.Interval_period == 0:
                dictmodel = {"m": model.state_dict()}
                torch.save(dictmodel,"model_data_{}.pth".format(epoch))
            writer.add_scalar("loss_echo", running_loss/total, epoch)
            writer.flush()
    print('Finished Training')

if __name__ == "__main__":
    main()