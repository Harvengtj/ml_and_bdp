import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Generator(nn.Module):
    """
    U-Net Generator for Image Colorization.
    Supports both regression (2 channels: a, b) and classification (N channels: colour bins).
    """
    def __init__(self, input_nc=1, output_nc=2, image_size=256, ngf=64, use_classification=False, num_bins=100):
        super().__init__()
        self.use_classification = use_classification
        self.num_bins = num_bins
        # Final output channels depend on the mode (regression vs classification)
        final_output_nc = num_bins if use_classification else output_nc
        
        num_downs = int(math.log2(image_size))
        # Build U-Net structure with skip connections
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, use_dropout=True)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block)
        
        # Outer-most layer determines the final output shape and activation
        self.model = UnetSkipConnectionBlock(final_output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, use_classification=use_classification)

    def forward(self, x):
        return self.model(x)

class UnetSkipConnectionBlock(nn.Module):
    """
    Defines a U-Net submodule with skip connection.
    X ------------------- (layer) ------------------- +
      |-- downsampling -- [submodule] -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, use_classification=False):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            # No Tanh for classification (we want raw logits for CrossEntropy)
            # Tanh is used for regression to squash outputs to [-1, 1]
            if use_classification:
                up = [uprelu, upconv]
            else:
                up = [uprelu, upconv, nn.Tanh()]
            model = [downconv] + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            model = [downrelu, downconv] + [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            model = [downrelu, downconv, downnorm] + [submodule] + [uprelu, upconv, upnorm]
            if use_dropout:
                model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # Add skip connection (concatenate across channels)
            return torch.cat([x, self.model(x)], 1)

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Classifies small patches of the image as real or fake.
    Input NC usually 3 (L + ab).
    """
    def __init__(self, input_nc=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)
        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        # Use sigmoid to output a probability map [0, 1]
        return torch.sigmoid(self.final(x))
