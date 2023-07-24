import torch
from torch import nn, Tensor
import torch.nn.functional as F
# from torchinfo import summary as Model_Summary
import torch.optim as optim
from typing import Optional
# import torchinfo
# from torchsummary import summary

# Convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding='same'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        # could also use default momentum and eps values for batch norm
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        
    def forward(self, x):
        return F.relu(self.batch_norm(self.conv(x)))
    
    # Inception blocks
class InceptionBlockA(nn.Module):
    def __init__(self, 
                in_channels: int,
                pool_features: int
            ):
        super(InceptionBlockA, self).__init__()
        self.branch1x1 = ConvBlock(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = ConvBlock(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = ConvBlock(48, 64, kernel_size=5)

        self.branch3x3dbl_1 = ConvBlock(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = ConvBlock(64, 96, kernel_size=3)
        self.branch3x3dbl_3 = ConvBlock(96, 96, kernel_size=3)

        self.branch_pool = ConvBlock(in_channels, pool_features, kernel_size=1)
    
    def forward(self, x: Tensor):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        branches = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(branches, 1)

class InceptionBlockB(nn.Module):
    def __init__(self, 
                in_channels: int,
            ):
        super(InceptionBlockB, self).__init__()

        self.branch3x3 = ConvBlock(in_channels, 384, kernel_size=3, stride=2, padding='valid')
        self.branch3x3dbl_1 = ConvBlock(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = ConvBlock(64, 96, kernel_size=3)
        self.branch3x3dbl_3 = ConvBlock(96, 96, kernel_size=3, stride=2, padding='valid')
    
    def forward(self, x: Tensor):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        branches = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(branches, 1)

class InceptionBlockC(nn.Module):
    def __init__(self, 
                in_channels: int,
                channels_7x7: int,
            ):
        super(InceptionBlockC, self).__init__()

        self.branch1x1 = ConvBlock(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = ConvBlock(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = ConvBlock(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = ConvBlock(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        # self.branch7x7_2 = ConvBlock(c7, c7, kernel_size=(1, 7))
        # self.branch7x7_3 = ConvBlock(c7, 192, kernel_size=(7, 1))

        self.branch7x7dbl_1 = ConvBlock(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = ConvBlock(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = ConvBlock(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = ConvBlock(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = ConvBlock(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        # self.branch7x7dbl_2 = ConvBlock(c7, c7, kernel_size=(7, 1))
        # self.branch7x7dbl_3 = ConvBlock(c7, c7, kernel_size=(1, 7))
        # self.branch7x7dbl_4 = ConvBlock(c7, c7, kernel_size=(7, 1))
        # self.branch7x7dbl_5 = ConvBlock(c7, 192, kernel_size=(1, 7))

        self.branch_pool = ConvBlock(in_channels, 192, kernel_size=1)
    
    def forward(self, x: Tensor):
        
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        branches = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(branches, 1)

class InceptionBlockD(nn.Module):
    def __init__(self, 
                in_channels: int,
            ):
        super(InceptionBlockD, self).__init__()

        self.branch3x3_1 = ConvBlock(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = ConvBlock(192, 320, kernel_size=3, stride=2, padding='valid')

        self.branch7x7x3_1 = ConvBlock(in_channels, 192, kernel_size=1)
        # self.branch7x7x3_2 = ConvBlock(192, 192, kernel_size=(1, 7), padding=(0, 3))
        # self.branch7x7x3_3 = ConvBlock(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_2 = ConvBlock(192, 192, kernel_size=(1, 7))
        self.branch7x7x3_3 = ConvBlock(192, 192, kernel_size=(7, 1))
        self.branch7x7x3_4 = ConvBlock(192, 192, kernel_size=3, stride=2, padding='valid')
    
    def forward(self, x: Tensor):

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        branches = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(branches, 1)

class InceptionBlockE(nn.Module):
    def __init__(self, 
                in_channels: int,
            ):
        super(InceptionBlockE, self).__init__()
        
        self.branch1x1 = ConvBlock(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = ConvBlock(in_channels, 384, kernel_size=1)
        # self.branch3x3_2a = ConvBlock(384, 384, kernel_size=(1, 3), padding=(0, 1))
        # self.branch3x3_2b = ConvBlock(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3_2a = ConvBlock(384, 384, kernel_size=(1, 3))
        self.branch3x3_2b = ConvBlock(384, 384, kernel_size=(3, 1))

        self.branch3x3dbl_1 = ConvBlock(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = ConvBlock(448, 384, kernel_size=3)
        # self.branch3x3dbl_3a = ConvBlock(384, 384, kernel_size=(1, 3), padding=(0, 1))
        # self.branch3x3dbl_3b = ConvBlock(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_3a = ConvBlock(384, 384, kernel_size=(1, 3))
        self.branch3x3dbl_3b = ConvBlock(384, 384, kernel_size=(3, 1))

        self.branch_pool = ConvBlock(in_channels, 192, kernel_size=1)
    
    def forward(self, x: Tensor):

        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        branches = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(branches, 1)

# Auxiliary heads
class InceptionAux(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 num_classes: int, 
            ):
        super(InceptionAux, self).__init__()
        self.conv0 = ConvBlock(in_channels, 128, kernel_size=1)
        self.conv1 = ConvBlock(128, 768, kernel_size=5, padding='valid')
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001
    
    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3, padding=0)
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, (1 ,1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
# Define model
class InceptionV3(nn.Module):
    def __init__(self, 
                 num_classes:int=2,
                 aux_logits: bool=True
                 ):
        super(InceptionV3, self).__init__()

        self.aux_logits = aux_logits
        
        # Initial convolutional and pooling layers
        self.conv0 = ConvBlock(3, 32, kernel_size=3, stride=2, padding=0)
        self.conv1 = ConvBlock(32, 32, kernel_size=3)
        self.conv2 = ConvBlock(32, 64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = ConvBlock(64, 80, kernel_size=1)
        self.conv4 = ConvBlock(80, 192, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Inception blocks
        self.inception_mixed_0 = InceptionBlockA(192, pool_features=32)
        self.inception_mixed_1 = InceptionBlockA(256, pool_features=64)
        self.inception_mixed_2 = InceptionBlockA(288, pool_features=64)
        self.inception_mixed_3 = InceptionBlockB(288)
        self.inception_mixed_4 = InceptionBlockC(768, channels_7x7=128)
        self.inception_mixed_5 = InceptionBlockC(768, channels_7x7=160)
        self.inception_mixed_6 = InceptionBlockC(768, channels_7x7=160)
        self.inception_mixed_7 = InceptionBlockC(768, channels_7x7=192)
        self.inception_mixed_8 = InceptionBlockD(768)
        self.inception_mixed_9 = InceptionBlockE(1280)
        self.inception_mixed_10 = InceptionBlockE(2048)
        
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)

        # Final layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.8)
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        # Initial layers
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        # Inception blocks
        x = self.inception_mixed_0(x)
        x = self.inception_mixed_1(x)
        x = self.inception_mixed_2(x)
        x = self.inception_mixed_3(x)
        x = self.inception_mixed_4(x)
        x = self.inception_mixed_5(x)
        x = self.inception_mixed_6(x)
        x = self.inception_mixed_7(x)
        # Auxiliary heads
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # more Inception
        x = self.inception_mixed_8(x)
        x = self.inception_mixed_9(x)
        x = self.inception_mixed_10(x)

        x = self.avg_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, aux
    
def main():
    model = InceptionV3()

    # Printing model architecture summary
    # print(summary(model, input_size=(3, 299, 299)))
    # print("\n TORCHINFO SUMMARY \n")
    # print(torchinfo.summary(model, (3, 299, 299), batch_dim=0, col_names=('input_size', 'output_size', 'num_params', 'kernel_size'), verbose=0))

if __name__=='__main__':
    main()