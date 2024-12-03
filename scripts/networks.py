import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from diffusers import VQModel

class Maxtractor(nn.Module):
    def __init__(self, device="cuda", train_enc=True, dropout_prob = 0.3):
        super(Maxtractor, self).__init__()
        # Define separate ResNet for RGB and depth
        #self.rgb_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        #self.depth_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model = Classifier_Max(device=device, train_enc=train_enc)

        self.cnn = nn.Sequential(
            # The input is [batch_size, 2048, 1]
            nn.Conv1d(in_channels=1000, out_channels=256, kernel_size=3, padding=1),  # Output shape: [batch_size, 1024, 1]
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),   # Output shape: [batch_size, 512, 1]
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),    # Output shape: [batch_size, 365, 1]
            nn.AdaptiveAvgPool1d(1),  # This ensures the output is [batch_size, 365]
            nn.Flatten()  # Flatten the output to [batch_size, 365]
        )


        # Common final fully connected layer
        self.fc = nn.Linear(128, 2)  # Concatenating the RGB and depth features
        self.dropout = nn.Dropout(p=dropout_prob)

        self.model.to(device)

        if train_enc:
            self.model.train()
        else:
            self.model.eval()

    def forward(self, spec_input):
        features = self.model(spec_input)
        features = self.cnn(features.unsqueeze(2))
        features = self.dropout(features)

        x = self.fc(features)
        #x = torch.sigmoid(x) * 200  # Scale to range [0, 200]
        #x = torch.nn.functional.relu(x)  # Scale to range [0, 200]
        return x


class Classifier_Max(nn.Module):
    """
    Load encoder from pre-trained ResNet50 (Places365 CNN) model.
    """
    def __init__(self, device="cuda", train_enc=True):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1), # [batch, 3, 512, 512]
            nn.ReLU(),
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [batch, 64, 256, 256]
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=2, padding=1),  # [batch, 3, 128, 128]
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)  # [batch, 3, 224, 224]
        )
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        #self.model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        self.model.fc = nn.Identity()

        if train_enc:
            self.model.train()
        else:
            self.model.eval()

    def forward(self, x):
        #print(f"Shape of x after channel_conv: {x.shape}")
        x = self.downsample(x)
        x = self.model.forward(x)
        #print(f"Shape of x after model.forward: {x.shape}")
        return x
    
class Encoder(nn.Module):
    def __init__(self, embedding_dim=3):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        #print(f"Shape of x before Quantizer: {x.shape}")
        return x
    
class VQQuantizer(VQModel):
    def __init__(self, embedding_dim=3, num_embeddings=128, in_channels=3, out_channels=3, commitment_cost=0.25):
        super(VQQuantizer, self).__init__()
        self.quantizer = VQModel(in_channels, out_channels, latent_channels=embedding_dim, num_vq_embeddings=num_embeddings, vq_embed_dim=embedding_dim)
        self.commitment_cost = commitment_cost

    def forward(self, x, return_dict=True):
        #quantized, _, quantization_loss = self.quantizer(x)
        VQEncoderOutput = self.quantizer(x)
        quantized = VQEncoderOutput['sample']
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        print(f"e_latent_loss = {e_latent_loss}")
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)
        print(f"q_latent_loss = {q_latent_loss}")
        #quantization_loss = [q_latent_loss, self.commitment_cost * e_latent_loss]

        #return quantized, quantization_loss
        return quantized, q_latent_loss, self.commitment_cost * e_latent_loss

class Decoder(nn.Module):
    def __init__(self, embedding_dim=3):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 2, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.tanh(self.conv3(x))
        #print(f"Shape of x after Decoder: {x.shape}")
        return x


class VQVAE(nn.Module):
    def __init__(self, embedding_dim=3, num_embeddings=128, in_channels=3, out_channels=3, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(embedding_dim)
        #self.quantizer = VQModel(num_vector_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.quantizer = VQQuantizer(embedding_dim, num_embeddings, in_channels, out_channels, commitment_cost=commitment_cost)
        self.decoder = Decoder(embedding_dim)

    def forward(self, x):
        #print(f"Shape of x right before Quantizer: {x.shape}")
        z_e = self.encoder(x)                   # Encode input to latent representation
        quantized, q_loss_1, q_loss_2 = self.quantizer(z_e)  # Quantize with diffusers.VQModel
        x_recon = self.decoder(quantized)       # Decode quantized representation to reconstruct input
        return x_recon, q_loss_1, q_loss_2

def main():
    # Model parameters
    embedding_dim = 3
    num_embeddings = 128
    batch_size = 4
    input_channels = 3  # Complex spectrogram with real and imaginary parts

    # Initialize model
    model = VQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings, 
                  in_channels=input_channels, out_channels=3).to('cuda')

    # Create a sample batch of complex spectrograms with shape [batch, channels, height, width]
    input_data = torch.randn(batch_size, 1, 512, 512).to('cuda')

    # Run the model
    reconstructed, quantization_loss = model(input_data)

    # Output shapes and quantization loss
    print(f"Input shape: {input_data.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Quantization loss: {quantization_loss.item()}")

if __name__ == "__main__":
    main()