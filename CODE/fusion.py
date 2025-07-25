## Project: LiDAR & Camera Fusion Depth Completion
### dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

class KITTIDepthDataset(Dataset):
    """
    KITTI depth completion dataset with paper-aligned preprocessing:
    - Bottom crop to 928x256 (Sec III-E)
    - Random sparse sampling (200 pts) per training epoch
    - Precomputed sparse for val/test
    """
    def __init__(self, rgb_dir, lidar_dir, gt_dir, transform=None, mode='train', num_samples=200):
        self.rgb_dir = rgb_dir
        self.lidar_dir = lidar_dir
        self.gt_dir = gt_dir
        self.ids = sorted(os.listdir(rgb_dir))
        self.transform = transform
        self.mode = mode
        self.num_samples = num_samples

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        rgb = cv2.imread(os.path.join(self.rgb_dir, img_id))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        gt = cv2.imread(os.path.join(self.gt_dir, img_id), cv2.IMREAD_UNCHANGED).astype(np.float32)

        # KITTI crop (928x256)
        h, w = rgb.shape[:2]
        crop_h, crop_w = 256, 928
        rgb = rgb[h-crop_h:h, (w-crop_w)//2:(w+crop_w)//2]
        gt = gt[h-crop_h:h, (w-crop_w)//2:(w+crop_w)//2]

        # Sparse depth generation
        if self.mode == 'train':
            mask = (gt > 0).flatten()
            valid_idx = np.where(mask)[0]
            sample_idx = np.random.choice(valid_idx, self.num_samples, replace=False)
            sparse = np.zeros_like(gt)
            sparse.flat[sample_idx] = gt.flat[sample_idx]
        else:
            sparse_path = os.path.join(self.lidar_dir, img_id.replace('.png', '.npy'))
            sparse = np.load(sparse_path)
            sparse = sparse[h-crop_h:h, (w-crop_w)//2:(w+crop_w)//2]

        if self.transform:
            rgb, sparse, gt = self.transform(rgb, sparse, gt)

        # Fuse channels: RGB + sparse
        fused = np.concatenate([rgb.transpose(2,0,1), sparse[None]], axis=0)
        return torch.from_numpy(fused), torch.from_numpy(gt)

# DataLoader factory

def get_loader(rgb_dir, lidar_dir, gt_dir, batch_size=8, shuffle=True, **kwargs):
    ds = KITTIDepthDataset(rgb_dir, lidar_dir, gt_dir, mode=kwargs.get('mode','train'), num_samples=kwargs.get('num_samples',200))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4)

import torch
import torch.nn as nn
import torchvision.models as models

class ResidualUpProjection(nn.Module):
    """Residual Up-Projection Block (Fig.3)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3,
                                         stride=2, padding=1, output_padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        up = self.deconv(out)
        sc = self.conv_sc(self.upsample(x))
        return self.relu(up + sc)

class DepthCompletionNet(nn.Module):
    """ResNet-50 encoder + RUB decoder"""
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        # Adapt first conv for 4-channel input
        self.encoder = nn.Sequential(
            nn.Conv2d(4,64,kernel_size=7,stride=2,padding=3,bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        # Decoder RUBs with skip concatenation
        self.up4 = ResidualUpProjection(2048, 512)
        self.up3 = ResidualUpProjection(512+1024, 256)
        self.up2 = ResidualUpProjection(256+512, 128)
        self.up1 = ResidualUpProjection(128+256, 64)
        self.out_conv = nn.Conv2d(64,1,kernel_size=3,padding=1)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x0 = self.encoder[0:5](x)  # 1/4
        x1 = self.encoder[5](x0)   # 1/8
        x2 = self.encoder[6](x1)   # 1/16
        x3 = self.encoder[7](x2)   # 1/32

        d4 = self.up4(x3)                          # →1/16
        d3 = self.up3(torch.cat([d4,x2],1))       # →1/8
        d2 = self.up2(torch.cat([d3,x1],1))       # →1/4
        d1 = self.up1(torch.cat([d2,x0],1))       # →1/2

        out = self.final_up(self.out_conv(d1))    # →1/1
        return out


### metrics.py

import torch

def rmse_metric(pred, gt, mask=None):
    if mask is None: mask = gt>0
    diff = (pred-gt)[mask]
    return torch.sqrt((diff**2).mean())

def rel_metric(pred, gt, mask=None):
    if mask is None: mask = gt>0
    diff = torch.abs(pred-gt)[mask] / gt[mask]
    return diff.mean()

def delta_metric(pred, gt, mask=None, threshold=1.25):
    """Percentage of pixels where max(pred/gt,gt/pred)<threshold"""
    if mask is None: mask = gt>0
    p=pred[mask]; g=gt[mask]
    ratio = torch.max(p/g, g/p)
    return (ratio<threshold).float().mean()


### train.py

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset import get_loader
from model import DepthCompletionNet
from metrics import rmse_metric, rel_metric, delta_metric
from utils import save_checkpoint, init_logger

def berhu_loss(pred, gt, mask):
    diff = pred[mask] - gt[mask]
    abs_diff = diff.abs()
    c = 0.2 * abs_diff.max()
    l1 = abs_diff
    l2 = (diff**2 + c**2) / (2*c)
    return torch.where(abs_diff<=c, l1, l2).mean()

if __name__ == '__main__':
    writer = SummaryWriter('runs/depth_completion')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DepthCompletionNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    loader = get_loader('rgb/','lidar/','gt/', batch_size=8, shuffle=True, mode='train')

    for epoch in range(20):
        model.train()
        for i,(fused,gt) in enumerate(loader):
            fused,gt = fused.to(device), gt.to(device)
            pred = model(fused)
            mask = gt>0
            loss = berhu_loss(pred, gt, mask)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            if i%10==0:
                rmse = rmse_metric(pred,gt,mask)
                rel  = rel_metric(pred,gt,mask)
                d1   = delta_metric(pred,gt,mask,1.25)
                d2   = delta_metric(pred,gt,mask,1.25**2)
                d3   = delta_metric(pred,gt,mask,1.25**3)
                global_step = epoch*len(loader)+i
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('RMSE/train', rmse.item(), global_step)
                writer.add_scalar('REL/train', rel.item(), global_step)
                writer.add_scalar('Delta1/train', d1.item(), global_step)
                writer.add_scalar('Delta2/train', d2.item(), global_step)
                writer.add_scalar('Delta3/train', d3.item(), global_step)
                print(f"Epoch{epoch} Iter{i} Loss{loss:.4f} RMSE{rmse:.4f} D1{d1:.3f}")

        save_checkpoint(model, optimizer, epoch, 'checkpoints/')
