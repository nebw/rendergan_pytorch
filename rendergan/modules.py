import torch 
from torch import nn

from attention import SelfAttn
from filters import GaussianHighpass, GaussianSmoothing
from spectral import SpectralNorm

class UpsampleConv2d(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size, padding=1, scale_factor=2):
       super(UpsampleConv2d, self).__init__()
       self.conv = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
       self.scale_factor = scale_factor
       
   @property
   def weight(self):
       return self.conv.weight
       
   def forward(self, x):        
       x = self.conv(x)
       x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, 
                                           mode='bilinear', align_corners=False)
       return x


class Generator(nn.Module):
    """Generator."""

    def __init__(self, image_size=64, z_dim=64, conv_dim=64):
        super(Generator, self).__init__()
        
        self.downblocks = nn.ModuleList([
            nn.Sequential(
                SpectralNorm(nn.Conv2d(1 + 4 + z_dim, conv_dim, 3, 1, 1)),
                nn.BatchNorm2d(conv_dim),
                nn.ReLU(),
                SpectralNorm(nn.Conv2d(conv_dim, conv_dim * 2, 3, 2, 1)),
                nn.BatchNorm2d(conv_dim * 2),
                nn.ReLU()
            ),
            nn.Sequential(
                SpectralNorm(nn.Conv2d(conv_dim * 2, conv_dim * 2, 3, 1, 1)),
                nn.BatchNorm2d(conv_dim * 2),
                nn.ReLU(),
                SpectralNorm(nn.Conv2d(conv_dim * 2, conv_dim * 4, 3, 2, 1)),
                nn.BatchNorm2d(conv_dim * 4),
                nn.ReLU(),
            ),
            nn.Sequential(
                SpectralNorm(nn.Conv2d(conv_dim * 4, conv_dim * 4, 3, 1, 1)),
                nn.BatchNorm2d(conv_dim * 4),
                nn.ReLU(),
                SpectralNorm(nn.Conv2d(conv_dim * 4, conv_dim * 8, 3, 2, 1)),
                nn.BatchNorm2d(conv_dim * 8),
                nn.ReLU(),
            ),
            nn.Sequential(
                SpectralNorm(nn.Conv2d(conv_dim * 8, conv_dim * 8, 3, 1, 1)),
                nn.BatchNorm2d(conv_dim * 8),
                nn.ReLU(),
                SpectralNorm(nn.Conv2d(conv_dim * 8, conv_dim * 16, 3, 2, 1)),
                nn.BatchNorm2d(conv_dim * 16),
                nn.ReLU(),
                SelfAttn(conv_dim * 16, 'relu'),
            )
        ])
        
        self.upblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(conv_dim * 16, conv_dim * 8 * 2 ** 2, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(conv_dim * 8),
                nn.ReLU(),
                SelfAttn(conv_dim * 8, 'relu'),
            ),
            nn.Sequential(
                nn.Conv2d(conv_dim * 16, conv_dim * 8, 3, 1, 1),
                nn.BatchNorm2d(conv_dim * 8),
                nn.ReLU(),
                nn.Conv2d(conv_dim * 8, conv_dim * 4 * 2 ** 2, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(conv_dim * 4),
                nn.ReLU(),
                SelfAttn(conv_dim * 4, 'relu'),
            ),
            nn.Sequential(
                nn.Conv2d(conv_dim * 8, conv_dim * 4, 3, 1, 1),
                nn.BatchNorm2d(conv_dim * 4),
                nn.ReLU(),
                nn.Conv2d(conv_dim * 4, conv_dim * 2 * 2 ** 2, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(conv_dim * 2),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(conv_dim * 4, conv_dim * 2, 3, 1, 1),
                nn.BatchNorm2d(conv_dim * 2),
                nn.ReLU(),
                nn.Conv2d(conv_dim * 2, conv_dim * 1 * 2 ** 2, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(conv_dim * 1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(conv_dim + 1 + 4 + z_dim, conv_dim * 1, 3, 1, 1),
                nn.BatchNorm2d(conv_dim * 1),
                nn.ReLU(),
                nn.Conv2d(conv_dim * 1, conv_dim * 1, 3, 1, 1),
                nn.BatchNorm2d(conv_dim * 1),
                nn.ReLU(),
                nn.Conv2d(conv_dim, 6, 3, 1, 1),
                nn.Tanh()
            )
        ])
        
        self.highpass = nn.Sequential(
            GaussianHighpass(1, 33, 2),
            GaussianHighpass(1, 33, 2),
            GaussianHighpass(1, 33, 2),
        )
        
        self.bg_mask_smoothing = GaussianSmoothing(1, 33, 1e-16, sigma_learnable=True, max_sigma=2)
        self.lighting_mask_smoothing = GaussianSmoothing(1, 33, 1e-16, sigma_learnable=True, max_sigma=2)
        
        self.smoothing = GaussianSmoothing(1, 33, 1e-16, sigma_learnable=True, max_sigma=2)
        
        self.lighting_blur = nn.Sequential(
            GaussianSmoothing(1, 33, 0.999, sigma_learnable=False, max_sigma=4),
        )  
        
        self.offsets = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(self, x, z):
        x_ = x[:, 0, :, :][:, None, :, :]
        
        black_offset = torch.sigmoid(self.offsets[0])
        white_offset = torch.sigmoid(self.offsets[1])
        
        bg = self.bg_mask_smoothing((x[:, 2, :, :][:, None, :, :] + 1) / 2)
        black = self.lighting_mask_smoothing((x[:, 3, :, :][:, None, :, :] + 1) / 2)
        white = self.lighting_mask_smoothing((x[:, 4, :, :][:, None, :, :] + 1) / 2)
                
        z = z.view(z.size(0), z.size(1), 1, 1)
        z = z.repeat(1, 1, 64, 64)
        x = torch.cat((x, z), dim=1)
        
        downs = [x]
        for mod in self.downblocks:
            x = mod(x)
            downs.append(x)
            
        x = self.upblocks[0](x)
        for dx, mod in zip(downs[:-1][::-1], self.upblocks[1:]):
            x = mod(torch.cat((x, dx), dim=1))
            
        smoothed_mask = self.smoothing(x_)
        background = bg * x[:, 0][:, None]
        
        foreground_highpass = (1 - bg) * self.highpass(x[:, 1][:, None] * 2)
        
        lighting_black = black * (((self.lighting_blur(x[:, 3, :, :][:, None, :, :]) + 1) / 2) * (0.9 - black_offset))
        lighting_white = white * (((self.lighting_blur(x[:, 4, :, :][:, None, :, :]) + 1) / 2) * (0.9 - white_offset) * (-1))
        lighting = self.lighting_blur(x[:, 5, :, :][:, None, :, :]) * 0.3
        
        generated_images = smoothed_mask + \
            background + \
            foreground_highpass + \
            lighting_black + lighting_white + lighting + \
            black * black_offset + white * white_offset
        generated_images_clamp = torch.clamp(generated_images, -1, 1)
        
        return generated_images, generated_images_clamp, (smoothed_mask, background, foreground_highpass, lighting_black, lighting_white, lighting)
            
    
class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(1, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        
        self.attn1 = SelfAttn(curr_dim, 'relu')

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim * 2
            
        self.attn2 = SelfAttn(curr_dim, 'relu')
            
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.attn1(out)
        out = self.l4(out)
        out = self.attn2(out)
        out = self.last(out)

        return out.squeeze()


class ParameterGenerator(nn.Module):
    def __init__(self, z_dim, label_dim, num_hidden):
        super(ParameterGenerator, self).__init__()

        self.gen = nn.Sequential(
            nn.Linear(z_dim, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, label_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


class Simulator(nn.Module):
    def __init__(self, label_dim=21, conv_dim=8):
        super(Simulator, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(label_dim, conv_dim * 4 * 4 ** 2, 1, 1, 0),
            nn.PixelShuffle(4),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(),
            
            nn.Conv2d(conv_dim * 4, conv_dim * 4, 3, 1, 1),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(),
            
            nn.Conv2d(conv_dim * 4, conv_dim * 4, 3, 1, 1),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(),
            
            nn.Conv2d(conv_dim * 4, conv_dim * 2, 3, 1, 1),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(),
            
            nn.Conv2d(conv_dim * 2, conv_dim * 2 * 4 ** 2, 1, 1, 0),
            nn.PixelShuffle(4),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(),
            
            nn.Conv2d(conv_dim * 2, conv_dim * 1, 3, 1, 1),
            nn.BatchNorm2d(conv_dim * 1),
            nn.ReLU(),
            
            nn.Conv2d(conv_dim * 1, conv_dim * 1 * 4 ** 2, 1, 1, 0),
            nn.PixelShuffle(4),
            nn.BatchNorm2d(conv_dim * 1),
            nn.ReLU(),
            
            nn.Conv2d(conv_dim * 1, conv_dim * 1, 3, 1, 1),
            nn.BatchNorm2d(conv_dim * 1),
            nn.ReLU(),
            
            nn.Conv2d(conv_dim * 1, 5, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, l):
        l = l.view(l.size(0), l.size(1), 1, 1)
        x = self.conv(l)
        
        return x