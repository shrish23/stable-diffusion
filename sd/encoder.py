import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

# Encoder reduces the dimension of the data
# at initial the image will have 3 channels and with each convolution we reduce the size of the image
# but at the same time we increase the number of channels (each pixel will have more features)


class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # (Batch size, channels, height, width) -> (Batch size, 128, height, width)
            nn.Conv2d(3,128, kernel_size=3, padding=1),

            # Residual Block: (Batch size, 128, height, width) -> (Batch size, 128, height, width)
            VAE_ResidualBlock(128, 128),# does not change the size of the image
            VAE_ResidualBlock(128, 128),# does not change the size of the image

            # The size now changes from (Batch size, 128, height, width) -> (Batch size, 128, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # Residual Block: (Batch size, 128, height/2, width/2) -> (Batch size, 256, height/2, width/2)
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), # (Batch size, 256, height/4, width/4)

            # Residual Block: (Batch size, 256, height/4, width/4) -> (Batch size, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), # (Batch size, 512, height/8, width/8)

            # Residual Block: (Batch size, 512, height/8, width/8) -> (Batch size, 1024, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # Attention Block: (Batch size, 512, height/8, width/8) -> (Batch size, 512, height/8, width/8)
            VAE_AttentionBlock(512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 

            # Group normalization
            nn.GroupNorm(32,512),

            nn.SiLU(), # Swish activation function

            # Bottleneck of the encoder
            nn.Conv2d(512, 8 , kernel_size=3, padding=1), # (Batch size, 8, height/8, width/8)

            nn.Conv2d(8, 8, kernel_size=1, padding=0) # (Batch size, 8, height/8, width/8)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        #x: (Batch size, channels, height, width)
        #noise: (Batch size, out_channels, height/8, width/8)
        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                # padding_left, padding_right, padding_top, padding_bottom
                x = F.pad(x, (0,1,0,1))# layer of pixels to the right and bottom
            x = module(x)
        
        # (Batch_size, 8, height/8, width/8) -> two tensors of shape (Batch_size, 4, height/8, width/8)
        mean, log_var = torch.chunk(x, 2, dim=1)# split the tensor in 2 parts along the channel dimension

        # clamping the values of the log variance: -30 <= log_var <= 20
        log_var = torch.clamp(log_var, -30, 20)

        variance = log_var.exp()# converting the log variance to variance

        # standard deviation
        stdev = variance.sqrt()

        # the latent space is a multivariate gaussian distribution
        # Now we can sample from the distribution
        # Z = N(0,1) -> N(mean, variance) =x
        # x = mean + Z * stdev
        x = mean + stdev * noise

        # scale the output by a constant
        x *= 0.18215

        return x