import torch
import torch.nn as nn
import numpy as np
from diffusers import VQModel
from .layers import PixelWiseNormLayer, MiniBatchAverageLayer, EqualizedLearningRateLayer # Components from PG-GAN

class Encoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(Encoder, self).__init__()
        # self.conv1 = nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1) # 512 -> 256
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # 512 -> 128
        # self.conv3 = nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1)

        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(2, 16, kernel_size=4, stride=2, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv2d(64, embedding_dim, kernel_size=3, stride=1, padding=1),
        #     PixelWiseNormLayer(),
        # )
        self.embedding_dim = embedding_dim
        self.build_model()
        
    def build_model(self): # [batch, 1, 512, 512]
        model = []
        model.append(nn.Conv2d(1, 512, kernel_size=4, stride=2, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)),
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)),
        model.append(PixelWiseNormLayer()) # [batch, 512, 256, 256]

        model.append(nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)),
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)),
        model.append(PixelWiseNormLayer()) # [batch, 256, 128, 128]

        model.append(nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)),
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)),
        model.append(PixelWiseNormLayer()) # [batch, 128, 64, 64]

        model.append(nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)),
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)),
        model.append(PixelWiseNormLayer()) # [batch, 64, 32, 32]

        model.append(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)),
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)),
        model.append(PixelWiseNormLayer()) # [batch, 32, 32, 32]
        
        model.append(nn.Conv2d(32, self.embedding_dim, kernel_size=1, stride=1, padding=0, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.Tanh()) # [bach, embedding_dim, 32, 32]

        self.model = nn.Sequential(*model)



    # def forward(self, x):
    #     x = torch.relu(self.conv1(x))
    #     x = torch.relu(self.conv2(x))
    #     x = self.conv3(x)
    #     #print(f"Shape of x before Quantizer: {x.shape}")
    #     return x

    def forward(self, x):
        return self.model(x)
    
class VQQuantizer(nn.Module):
    def __init__(self, embedding_dim=128, num_embeddings=128, in_channels=3, out_channels=3, commitment_cost=0.25):
        super(VQQuantizer, self).__init__()
        self.quantizer = VQModel(in_channels, out_channels, latent_channels=embedding_dim, num_vq_embeddings=num_embeddings, vq_embed_dim=embedding_dim)
        self.commitment_cost = commitment_cost

    def forward(self, x, return_dict=True):
        #quantized, _, quantization_loss = self.quantizer(x)
        VQEncoderOutput = self.quantizer(x)
        quantized = VQEncoderOutput['sample']
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)
        #quantization_loss = self.commitment_cost * e_latent_loss + q_latent_loss

        #return quantized, quantization_loss
        return quantized, q_latent_loss, self.commitment_cost * e_latent_loss

class Decoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(Decoder, self).__init__()
        # self.conv1 = nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        # self.conv3 = nn.ConvTranspose2d(128, 2, kernel_size=3, stride=1, padding=1)
        # self.tanh = nn.Tanh()
        self.embedding_dim = embedding_dim
        self.build_model()
        

        # self.deconv_layer = nn.Sequential(
        #     nn.ConvTranspose2d(embedding_dim, 64, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.ConvTranspose2d(16, 2, kernel_size=3, stride=1, padding=1),
        #     #nn.Tanh()
        # )

    def build_model(self): # [batch, embedding_dim, 64, 64]
        model = []
        model.append(nn.ConvTranspose2d(self.embedding_dim, 32, kernel_size=1, stride=1, padding=0, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)) #[batch, 32, 64, 64]

        model.append(nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)) # [batch, 64, 128, 128]

        model.append(nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)) # [batch, 128, 256, 256]

        model.append(nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)) # [batch, 256, 256, 256]

        model.append(MiniBatchAverageLayer()) # [batch, 129, 256, 256]
        model.append(nn.ConvTranspose2d(257, 512, kernel_size=3, stride=1, padding=1, bias=False)) 
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2)) # [batch, 512, 256, 256]

        model.append(nn.ConvTranspose2d(512, 1, kernel_size=4, stride=2, padding=1, bias=False)) # [batch, 1, 512, 512]

        self.model = nn.Sequential(*model)



    # def forward(self, x):
    #     x = torch.relu(self.conv1(x))
    #     x = torch.relu(self.conv2(x))
    #     x = self.tanh(self.conv3(x))
    #     #print(f"Shape of x after Decoder: {x.shape}")
    #     return x
    def forward(self, x):
        return self.model(x)


class VQVAE(nn.Module):
    def __init__(self, encoder, quantizer, decoder, embedding_dim=128, num_embeddings=128, in_channels=128, out_channels=128, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        #self.encoder = Encoder(embedding_dim)
        #self.quantizer = VQQuantizer(embedding_dim, num_embeddings, in_channels, out_channels, commitment_cost=commitment_cost)
        #self.decoder = Decoder(embedding_dim)
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

    def forward(self, x):
        #print(f"Shape of x right before Encoder: {x.shape}")
        z_e = self.encoder(x)                   # Encode input to latent representation
        #print(f"Shape of encoder output: {z_e.shape}")
        quantized, q_loss_1, q_loss_2 = self.quantizer(z_e)  # Quantize with diffusers.VQModel
        #print(f"Shape of quantized: {quantized.shape}")
        x_recon = self.decoder(quantized)       # Decode quantized representation to reconstruct input
        #print(f"Shape of x_recon: {x_recon.shape}")
        return x_recon, q_loss_1, q_loss_2

def main():
    # Model parameters
    embedding_dim = 3
    num_embeddings = 128
    batch_size = 2
    input_channels = 3  # Complex spectrogram with real and imaginary parts

    # Initialize model
    encoder = Encoder(embedding_dim=embedding_dim)
    quantizer = VQQuantizer(num_embeddings=num_embeddings, in_channels=embedding_dim, out_channels=embedding_dim)
    decoder = Decoder(embedding_dim=embedding_dim)
    model = VQVAE(encoder=encoder,quantizer=quantizer, decoder=decoder, embedding_dim=embedding_dim, num_embeddings=num_embeddings, 
                  in_channels=input_channels, out_channels=input_channels).to('cuda')
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    
    print(f"Memory allocated after model initialization: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"Total GPU memory reserved after model initialization: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

    # Create a sample batch of complex spectrograms with shape [batch, channels, height, width]
    input_data = torch.randn(batch_size, 1, 512, 512).to('cuda')
    print(f"Memory allocated after loading input data: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"Total GPU memory reserved after loading input data: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

    # Run the model
    with torch.no_grad():
        reconstructed, q_loss_1, q_loss_2 = model(input_data)

    # Output shapes and quantization loss
    print(f"Input shape: {input_data.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Q_Loss_1: {q_loss_1.item()}")
    print(f"Q_Loss_2: {q_loss_2.item()}")

        # Check memory after forward pass
    print(f"Memory allocated after forward pass: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"Total GPU memory reserved after forward pass: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    main()