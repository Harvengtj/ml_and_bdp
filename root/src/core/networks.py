import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math


def init_weights(m):
    """Initializes weights with a Gaussian distribution (mean = 0, std = 0.02) according to Pix2Pix/GAN standards."""
    
    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: # check if class is called 'Conv2d' or 'ConvTranspose2d'
        nn.init.normal_(m.weight.data, 0, 0.02)
    elif classname.find('BatchNorm2d') != -1: # check if class is called s'BatchNorm2d'
        nn.init.normal_(m.weight.data, 1, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

#===================================================================================================================================
#=== GENERATOR ===
#===================================================================================================================================
class Generator(nn.Module):
    """
    U-Net Generator for image colorization.
    Supports both regression (2 channels: a, b) and classification (N channels: colour bins).
    """
    def __init__(self, input_nc=1, output_nc=2, image_size=256, ngf=64, use_classification=False, num_bins=100):
        """
        Initialize U-Net Generator.
        We start from the deepest layer and we wrap the blocks towards the outer layer.
        
        Args:
            input_nc (int): Number of input channels at the very beginning.
            output_nc (int): Number of output channels at the very end.
            image_size (int): Size of the image.
            ngf (int): Number of Generator Filters.
            use_classification (bool): If True, use classification, else use regression.
            num_bins (int): Number of bins (number of output channels at the very end if classification).
        """
        
        super().__init__()
        self.use_classification = use_classification
        self.num_bins = num_bins
        # Final output channels depend on the mode (regression vs classification)
        final_output_nc = num_bins if use_classification else output_nc
        
        # Fixed depth
        num_downs = 7 
        
        # Build U-Net structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True) # create bottleneck
        for _ in range(num_downs - 5): # wrap recursively the blocks
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, use_dropout=True)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block)
        
        self.model = UnetSkipConnectionBlock(final_output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, use_classification=use_classification)

        self.apply(init_weights)

    def forward(self, x):
        # Use gradient checkpointing if in classification mode to save VRAM
        if self.use_classification and self.training and getattr(self, "use_checkpointing", True):
            return torch.utils.checkpoint.checkpoint(self.model, x, use_reentrant=False)
        return self.model(x)


#===================================================================================================================================
#=== SUB-MODULES FOR DISCRIMINATOR ===
#===================================================================================================================================
class UnetSkipConnectionBlock(nn.Module):
    """Defines a U-Net submodule with skip connection."""
    
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, use_classification=False):
        """
        Note that old convention was used: (leaky)relu, norm, conv.
        
        We need to see the network as Matryochka dolls.
        
        Args:
            outer_nc (int): Number of output channels for this block.
            inner_nc (int): Number of intermediate channels (after downsampling).
            input_nc (int, optional): Number of input channels. Defaults to outer_nc.
            submodule (nn.Module, optional): The nested U-Net block (deeper level).
            outermost (bool): If True, this is the outermost block (final output layer).
            innermost (bool): If True, this is the bottleneck block (deepest layer).
            norm_layer (nn.Module): Normalization layer to use (e.g., BatchNorm2d).
            use_dropout (bool): If True, applies dropout in intermediate blocks (regularization).
            use_classification (bool): If True, outputs raw logits (no Tanh activation),
                                    typically used with CrossEntropyLoss for classification tasks.
                                    If False, applies Tanh activation (commonly used for image regression to [-1, 1]).
        """
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc # ex: in the bottleneck

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # CASE 1: at the end of the network
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            # No Tanh for classification (we want raw logits for CrossEntropy)
            # Tanh is used for regression to squash outputs to [-1, 1]
            if use_classification:
                up = [uprelu, upconv] # no batch norm and output raw logits
            else:
                up = [uprelu, upconv, nn.Tanh()] # no batch norm and use of tanh
            model = [downconv] + [submodule] + up
            
        # CASE 2: at the bottleneck
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            model = [downrelu, downconv] + [uprelu, upconv, upnorm]
            
        # CASE 3: anywhere else
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            model = [downrelu, downconv, downnorm] + [submodule] + [uprelu, upconv, upnorm]
            if use_dropout:
                model += [nn.Dropout(0.5)] # randomly deactivates a portion of the neurons during training (only for expansive path)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # Add skip connection (concatenate across channels)
            return torch.cat([x, self.model(x)], 1)
        
        
#===================================================================================================================================
#=== DISCRIMINATOR ===
#===================================================================================================================================
class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Classifies small patches of the image as real or fake.
    
    Args:
        input_nc (int): Number of input channels.
    """
    def __init__(self, input_nc=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)
        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1) # returns a proba., so output_channels=1
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # Apply initialisation
        self.apply(init_weights)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        # Return raw logits for BCEWithLogitsLoss (no need for final activation function since included in BCE)
        return self.final(x)
