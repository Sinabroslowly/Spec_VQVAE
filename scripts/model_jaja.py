import torch
import torch.nn as nn
import numpy as np
from diffusers import VQModel
from .networks import VQVAE

class MultiVQVAE(VQVAE):
    def __init__(self, num_embeddings=64):
        super(MultiVQVAE, self).__init__()
        self.high_VQVAE = VQVAE(embedding_dim=3, num_embeddings=4*num_embeddings, in_channels=3, out_channels=3, commitment_cost=0.25)
        self.mid_VQVAE = VQVAE(embedding_dim=3, num_embeddings=2*num_embeddings, in_channels=3, out_channels=3, commitment_cost=0.25)
        self.low_VQVAE = VQVAE(embedding_dim=3, num_embeddings=num_embeddings, in_channels=3, out_channels=3, commitment_cost=0.25)

    def forward(self, x, return_dict=True):
        output_high, quant_loss_high = self.high_VQVAE(x)
        output_mid, quant_loss_mid = self.mid_VQVAE(x)
        output_low, quant_loss_low = self.low_VQVAE(x)

        return torch.stack([output_high, output_mid, output_low], dim=4), torch.stack([quant_loss_high, quant_loss_mid, quant_loss_low], dim=0)

def main():
    # Model parameters
    embedding_dim = 3
    num_embeddings = 32
    batch_size = 2
    input_channels = 3  # Complex spectrogram with real and imaginary parts

    # Clear any existing cached memory
    torch.cuda.empty_cache()

    # Initialize model and move it to GPU
    model = MultiVQVAE(num_embeddings=num_embeddings).to('cuda')
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    
    print(f"Memory allocated after model initialization: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"Total GPU memory reserved after model initialization: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

    # Create a sample batch of complex spectrograms with shape [batch, channels, height, width]
    input_data = torch.randn(batch_size, 2, 512, 512).to('cuda')
    print(f"Memory allocated after loading input data: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"Total GPU memory reserved after loading input data: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

    # Run the model
    with torch.no_grad():
        reconstructed, q_loss_1, q_loss_2 = model(input_data)

    # Output shapes and quantization loss
    print(f"Input shape: {input_data.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"q_loss_1: {q_loss_1}")
    print(f"q_loss_2: {q_loss_2}")

    # Check memory after forward pass
    print(f"Memory allocated after forward pass: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"Total GPU memory reserved after forward pass: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

if __name__ == "__main__":
    main()
