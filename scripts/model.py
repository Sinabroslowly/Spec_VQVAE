import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from diffusers import VQModel
from .layers import PixelWiseNormLayer, MiniBatchAverageLayer, EqualizedLearningRateLayer # Components from PG-GAN


class VQVAE(nn.Module):
    def __init__(self, encoder, quantizer, decoder, embedding_dim=128, num_embeddings=128, in_channels=128, out_channels=128, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.in_channels: int = 1,
        self.out_channels: int = 1,
        

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