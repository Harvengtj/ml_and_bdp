import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#===========================================================================================================================
#=== GENERATOR ===
#===========================================================================================================================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder layers: Conv2d, BN, leaky-ReLu
        # Size output: H_out = (H_in - K + 2P) / S + 1 
        #              W_out = (W_in - K + 2P) / S + 1
        self.conv1_2 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1) # attention to size
        self.conv2_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv4_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv5_6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv6_7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv7_8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv8_9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        
        # Decoder layer: ConvTranspose2d, BN, ReLu
        # Size output: H_out = (H_in - 1) * S - 2 * P + K (+ output_padding)
        #              W_out = (W_in - 1) * S - 2 * P + K (+ output_padding)
        self.deconv9_8 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.deconv8_7 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.deconv7_6 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.deconv6_5 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv5_4 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv4_3 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv2_1 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.deconv1_0 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1) # 1x1 conv with tanh and stride 1

        # Batch norm 2D
        # no bn1_2
        self.bn2_3 = nn.BatchNorm2d(num_features=64)
        self.bn3_4 = nn.BatchNorm2d(num_features=128)
        self.bn4_5 = nn.BatchNorm2d(num_features=256)
        self.bn5_6 = nn.BatchNorm2d(num_features=512)
        self.bn6_7 = nn.BatchNorm2d(num_features=512)
        self.bn7_8 = nn.BatchNorm2d(num_features=512)
        self.bn8_9 = nn.BatchNorm2d(num_features=512)
        self.bn9_8 = nn.BatchNorm2d(num_features=512)
        self.bn8_7 = nn.BatchNorm2d(num_features=512)
        self.bn7_6 = nn.BatchNorm2d(num_features=512)
        self.bn6_5 = nn.BatchNorm2d(num_features=256)
        self.bn5_4 = nn.BatchNorm2d(num_features=128)
        self.bn4_3 = nn.BatchNorm2d(num_features=64)
        self.bn3_2 = nn.BatchNorm2d(num_features=64)
        # no bn2_1
        # no bn1_0
        
   



    def forward(self, enc_x1):
        # --- Encoder ---
        # Layer 1-2: (1, 256, 256) --> (64, 256, 256)
        enc_x2 = F.leaky_relu(self.conv1_2(enc_x1), negative_slope=0.2) # no batch norm
        # Layer 2-3: (64, 256, 256) --> (64, 128, 128)
        enc_x3 = F.leaky_relu(self.bn2_3(self.conv2_3(enc_x2)), negative_slope=0.2)
        # Layer 3-4: (64, 128, 128) --> (128, 64, 64)
        enc_x4 = F.leaky_relu(self.bn3_4(self.conv3_4(enc_x3)), negative_slope=0.2)
        # Layer 4-5: (128, 64, 64) --> (256, 32,32)
        enc_x5 = F.leaky_relu(self.bn4_5(self.conv4_5(enc_x4)), negative_slope=0.2)
        # Layer 5-6: (256, 32,32) --> (512, 16, 16)
        enc_x6 = F.leaky_relu(self.bn5_6(self.conv5_6(enc_x5)), negative_slope=0.2)
        # Layer 6-7: (512, 16, 16) --> (512, 8, 8)
        enc_x7 = F.leaky_relu(self.bn6_7(self.conv6_7(enc_x6)), negative_slope=0.2)
        # Layer 7-8: (512, 8, 8) --> (512, 4, 4)
        enc_x8 = F.leaky_relu(self.bn7_8(self.conv7_8(enc_x7)), negative_slope=0.2)
        # Layer 8-9: (512, 4, 4) --> (512, 2, 2)
        enc_x9 = F.leaky_relu(self.bn8_9(self.conv8_9(enc_x8)), negative_slope=0.2)
        
        # --- Decoder ---
        # Layer 9-8: (512, 2, 2) --> (512, 4, 4)
        dec_x8 = F.relu(self.bn9_8(self.deconv9_8(enc_x9)))
        dec_x8 = torch.cat([enc_x8, dec_x8], dim=1)  # dim = 1 corresponds to channels
        # Layer 8-7: (1024, 4, 4) --> (512, 8, 8)
        dec_x7 = F.relu(self.bn8_7(self.deconv8_7(dec_x8)))
        dec_x7 = torch.cat([enc_x7, dec_x7], dim=1)
        # Layer 7-6: (1024, 8, 8) --> (512, 16, 16)
        dec_x6 = F.relu(self.bn7_6(self.deconv7_6(dec_x7)))
        dec_x6 = torch.cat([enc_x6, dec_x6], dim=1)
        # Layer 6-5: (1024, 16, 16) --> (256, 32, 32)
        dec_x5 = F.relu(self.bn6_5(self.deconv6_5(dec_x6)))
        dec_x5 = torch.cat([enc_x5, dec_x5], dim=1)
        # Layer 5-4: (512, 32, 32) --> (128, 64, 64)
        dec_x4 = F.relu(self.bn5_4(self.deconv5_4(dec_x5)))
        dec_x4 = torch.cat([enc_x4, dec_x4], dim=1)
        # Layer 4-3: (256, 64, 64) --> (64, 128, 128)
        dec_x3 = F.relu(self.bn4_3(self.deconv4_3(dec_x4)))
        dec_x3 = torch.cat([enc_x3, dec_x3], dim=1)
        # Layer 3-2: (128, 128, 128) --> (64, 256, 256)
        dec_x2 = F.relu(self.bn3_2(self.deconv3_2(dec_x3)))
        dec_x2 = torch.cat([enc_x2, dec_x2], dim=1)
        # Layer 2-1: (128, 256, 256) --> (3, 256, 256)
        dec_x1 = F.relu(self.deconv2_1(dec_x2)) # no batch norm
        # Layer 1-0: (3, 256, 256) --> (3, 256, 256)
        dec_x0 = torch.tanh(self.deconv1_0(dec_x1)) # no batch norm
        
        return dec_x0
    
    


#===========================================================================================================================
#=== DISCRIMINATOR ===
#===========================================================================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder layers: Conv2d, BN, leaky-ReLu
        # Size output: H_out = (H_in - K + 2P) / S + 1 
        #              W_out = (W_in - K + 2P) / S + 1
        self.conv1_2 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1) # attention to size
        self.conv3_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv4_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv5_6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.last = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, padding=1)
  
    

        # Batch norm 2D
        # no bn1_2
        self.bn3_4 = nn.BatchNorm2d(num_features=128)
        self.bn4_5 = nn.BatchNorm2d(num_features=256)
        self.bn5_6 = nn.BatchNorm2d(num_features=512)  
        
        
   



    def forward(self, enc_x1):
        # --- Encoder ---
        # Layer 1-2: (1, 256, 256) --> (64, 256, 256)
        enc_x2 = F.leaky_relu(self.conv1_2(enc_x1), negative_slope=0.2) # no batch norm
        # Layer 3-4: (64, 256, 256) --> (128, 128, 128)
        enc_x4 = F.leaky_relu(self.bn3_4(self.conv3_4(enc_x2)), negative_slope=0.2)
        # Layer 4-5: (128, 128, 128) --> (256, 64, 64)
        enc_x5 = F.leaky_relu(self.bn4_5(self.conv4_5(enc_x4)), negative_slope=0.2)
        # Layer 5-6: (256, 64, 64) --> (512, 32, 32)
        enc_x6 = F.leaky_relu(self.bn5_6(self.conv5_6(enc_x5)), negative_slope=0.2)
        # Last layer: (512, 32, 32) --> (1, 32, 32)
        last = torch.sigmoid(self.last(enc_x6), negative_slope=0.2)
        
        return last



def create_discriminator(self):
        kernels_dis = [
            (64, 2, 0),
            (128, 2, 0),
            (256, 2, 0),
            (512, 1, 0),
        ]

        return Discriminator('dis', kernels_dis, training=self.options.training)