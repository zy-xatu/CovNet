import time
import torch
import numpy as np
import torch.nn as nn
from thop import profile

class CovNet(nn.Module):
    def __init__(self):
        super(CovNet, self).__init__()
        self.cfe = CFE(3, 256)
        self.head = Head()
        self.stem_layer = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.stage_layer1 = Stage_layer(16, 32, 3, 2, 1, 1)
        self.stage_layer2 = Stage_layer(32, 64, 3, 2, 1, 1)
        self.stage_layer3 = Stage_layer(64, 64, 3, 2, 1, 1)
        self.d2t1 = D2TModule(64, 64, dilation=4)
        self.d2t2 = D2TModule(64, 32, dilation=4)
        self.d2t3 = D2TModule(32, 16, dilation=4)
        self.d2t4 = D2TModule(16, 3, dilation=4)
        
        # self.stage_layer = nn.ModuleList([nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        #                                   Stage_layer(16, 32, 3, 2, 1, 1),
        #                                   Stage_layer(32, 64, 3, 2, 1, 1),
        #                                   Stage_layer(64, 64, 3, 2, 1, 1)])
        # self.d2t = nn.ModuleList([D2TModule(64, 64, dilation=4),
        #                           D2TModule(64, 32, dilation=4),
        #                           D2TModule(32, 16, dilation=4),
        #                           D2TModule(16, 3, dilation=4)])
        # self.down_sample = []

    def forward(self, x1, x2):
        """
        x1:t image
        x2:t+1 image
        """
        x = self.cfe(x1, x2)
        x_stem_layer = self.stem_layer(x)
        x_stage_layer1 = self.stage_layer1(x_stem_layer)
        x_stage_layer2 = self.stage_layer2(x_stage_layer1)
        x_stage_layer3 = self.stage_layer3(x_stage_layer2)
        x_d2t1 = torch.add(self.d2t1(x_stage_layer3), x_stage_layer2)
        x_d2t2 = torch.add(self.d2t2(x_d2t1), x_stage_layer1)
        x_d2t3 = torch.add(self.d2t3(x_d2t2), x_stem_layer)
        x = self.d2t4(x_d2t3)
        output = self.head(x1, x).flatten(0)
        top_5 = torch.topk(output.flatten(0), 5, dim=0, largest=True, sorted=True, out=None)
        return output,top_5

class CFE(nn.Module):
    def __init__(self, in_c, out_c):
        super(CFE, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=16, stride=16, padding=0)
        self.conv2 = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):
        x_patch = self.conv1(x).flatten(2).unsqueeze(1)
        y_patch = self.conv1(y).flatten(2).unsqueeze(1)
        x_pixel = self.conv2(x)
        y_pixel = self.conv2(y)
        result_patch = (x_patch -torch.mean(x_patch)) @ ((y_patch-torch.mean(y_patch))).transpose(-1,-2) / (x_patch.shape[2] * x_patch.shape[3] - 1)
        result_pixel = (x_pixel -torch.mean(x_pixel)) @ ((y_pixel-torch.mean(y_pixel)).transpose(-1,-2)) / (x_pixel.shape[2] * x_pixel.shape[3] - 1)
        result = torch.cat((x_patch * result_patch, x_pixel*result_pixel, y_patch*result_patch, y_pixel*result_pixel), dim=1)
        return result

class Head(nn.Module):
    def __init__(self, kernel_size=1, stride=1, padding=0):
        super(Head, self).__init__()
        self.image_conv = nn.Conv2d(3, 1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(3, 1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_final = nn.Conv2d(69, 1, kernel_size=1, stride=1, padding=0)
        

    def forward(self, image, x):
        image = self.image_conv(image)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        y = torch.concat((x1, x2, x3, image), dim=1)
        return self.conv_final(y).squeeze()

class D2TModule(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=1, stride=1, padding=0, dilation=1, scale_factor=2):
        super(D2TModule, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor)
        self.attn = C2f(in_c, in_c)
        self.conv = ConvModule(in_c, out_c , kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    
    def forward(self, x):
        return self.conv(self.attn(self.upsample(x)))

class ConvModule(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding, dilation=1):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Stage_layer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=2, padding=1, dilation=1):
        super(Stage_layer, self).__init__()
        self.conv = ConvModule(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.attn = C2f(out_c, out_c)
    
    def forward(self, x):
        return self.attn(self.conv(x))

class C2f(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=1, stride=1, padding=0, deepen_factor = 1, deepth=6, e=0.5, add = True):
        super(C2f, self).__init__()
        self.c = int(out_c * e)
        self.deepth = int(deepen_factor * deepth)
        self.conv1 = ConvModule(in_c,
                          out_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = ConvModule((2 + deepth) * self.c, out_c,
                          kernel_size=kernel_size, stride=stride, padding=padding)
        self.darknet = nn.ModuleList(DarknetBottleneck(
            self.c, kernel_size=kernel_size, stride=stride, padding=0, add=add) for _ in range(deepth))

    def forward(self, x):
        y = list(self.conv1(x).split((self.c, self.c), 1))
        y.extend(darknet(y[-1]) for darknet in self.darknet)
        return self.conv2(torch.cat(y, 1))

class DarknetBottleneck(nn.Module):
    def __init__(self, in_c, kernel_size=3, stride=1, padding=1, e=0.5, add=True):
        super(DarknetBottleneck, self).__init__()
        self.c = int(e * in_c)
        self.add = add
        self.conv1 = ConvModule(in_c,
                          self.c,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding)
        self.conv2 = ConvModule(self.c,
                          in_c,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding)

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add == True else self.conv2(self.conv1(x))

if __name__ == '__main__':
    device = "cpu" if torch.cuda.is_available else "cpu"
    model = CovNet().to(device)
    image_t0 = torch.randn(1, 3, 256, 256).to(device)
    image_t1 = torch.randn(1, 3, 256, 256).to(device)
    start = time.perf_counter()
    x = model(image_t0, image_t1)
    end = time.perf_counter()
    flops, params = profile(model, inputs=(image_t0, image_t1))
    print("FPS:{}".format(1/(end-start)))
    print(f"FLOPs: {flops/1000000.0} M, Params: {params/1000000.0} M")
    # print(x)
    