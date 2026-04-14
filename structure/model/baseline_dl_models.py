import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineAdapter(nn.Module):
    """
    Data adapter: convert data from (Batch, Bands, Time, Channels)
    to the classic EEGNet / DeepConvNet format of (Batch, 1, Channels, Time).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: (Batch, Bands=55, Time=512, Channels=22)
        # 1. Sum across all frequency bands (reconstructing broadband signal from filter bank outputs) -> (Batch, Time, Channels)
        x = x.sum(dim=1)
        # 2. Rearrange dimensions -> (Batch, Channels, Time)
        x = x.permute(0, 2, 1)
        # 3. Add channel dimension (channel here corresponds to feature map channels in deep learning, set to 1) -> (Batch, 1, 22, 512)
        x = x.unsqueeze(1)
        return x


class EEGNet(nn.Module):
    """
    Classic EEGNet
    """
    def __init__(self, nb_classes=2, Chans=22, Samples=512, dropoutRate=0.5, 
                 kernLength=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        self.adapter = BaselineAdapter()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwise1 = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)

        # Block 2
        self.separable1_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 16 // 2), groups=F1 * D, bias=False)
        self.separable1_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)

        # Dynamically compute the flattened output dimension
        self.flatten_dim = self._calculate_out_dim(Chans, Samples)
        
        self.classifier = nn.Linear(self.flatten_dim, nb_classes)

    def _calculate_out_dim(self, chans, samples):
        # Construct a dummy tensor with shape (1, 1, Chans, Samples) to compute output dimension
        dummy = torch.zeros(1, 1, chans, samples)
        x = self.conv1(dummy)
        x = self.batchnorm1(x)
        x = self.depthwise1(x)
        x = self.batchnorm2(x)
        x = self.pooling1(x)
        x = self.separable1_depth(x)
        x = self.separable1_point(x)
        x = self.batchnorm3(x)
        x = self.pooling2(x)
        return x.numel() # numel() returns the total number of elements in the tensor

    def forward(self, x):
        x = self.adapter(x)
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise1(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        x = self.separable1_depth(x)
        x = self.separable1_point(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        x = x.reshape(-1, self.flatten_dim)
        out = self.classifier(x)
        return out, torch.tensor(0.0).to(x.device)


class DeepConvNet(nn.Module):
    """
    Classic DeepConvNet 
    """
    def __init__(self, nb_classes=2, Chans=22, Samples=512, dropoutRate=0.5):
        super(DeepConvNet, self).__init__()
        self.adapter = BaselineAdapter()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 25, (1, 10), bias=False)
        self.conv2 = nn.Conv2d(25, 25, (Chans, 1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(25)
        self.pooling1 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.dropout1 = nn.Dropout(dropoutRate)

        # Block 2
        self.conv3 = nn.Conv2d(25, 50, (1, 10), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(50)
        self.pooling2 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.dropout2 = nn.Dropout(dropoutRate)

        # Block 3
        self.conv4 = nn.Conv2d(50, 100, (1, 10), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(100)
        self.pooling3 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.dropout3 = nn.Dropout(dropoutRate)

        # Block 4
        self.conv5 = nn.Conv2d(100, 200, (1, 10), bias=False)
        self.batchnorm4 = nn.BatchNorm2d(200)
        self.pooling4 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.dropout4 = nn.Dropout(dropoutRate)

        # Dynamically compute the flattened output dimension
        self.flatten_dim = self._calculate_out_dim(Chans, Samples)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, nb_classes)
        )

    def _calculate_out_dim(self, chans, samples):
        # Construct a dummy tensor to compute output dimension
        dummy = torch.zeros(1, 1, chans, samples)
        x = self.conv1(dummy)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.pooling1(x)
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = self.pooling2(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.pooling3(x)
        x = self.conv5(x)
        x = self.batchnorm4(x)
        x = self.pooling4(x)
        return x.numel()

    def forward(self, x):
        x = self.adapter(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling3(x)
        x = self.dropout3(x)

        x = self.conv5(x)
        x = self.batchnorm4(x)
        x = F.elu(x)
        x = self.pooling4(x)
        x = self.dropout4(x)

        out = self.classifier(x)
        return out, torch.tensor(0.0).to(x.device)