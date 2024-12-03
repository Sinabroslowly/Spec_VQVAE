import numpy as np
import torch
import librosa
import torch.nn.functional as F

class STFT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._eps = 1e-8

    def transform(self, audio, encode_flag=False):
        m = librosa.stft(audio, n_fft=1024, hop_length=256)[:-1, :]
        real, imag = np.real(m), np.imag(m)
        if encode_flag:
            mag, phase = np.abs(m), np.angle(m)
            mag = np.log10(mag + self._eps)
            mag = (((mag - mag.min()) / (mag.max() - mag.min()) * 2) - 1) # Normalize magnitude values [-1, 1]
            phase = np.degrees(phase) # Convert phase values to degrees
            phase = ((phase - (-180)) / (180 - (-180)) * 2) - 1 # Normalize phase values [-180, 180] to [-1, 1]
            #print(f"Applying normalized magnitude and phase spectrogram: {mag.shape}, {phase.shape}")
            return (torch.FloatTensor if torch.cuda.is_available() else torch.Tensor)(mag).unsqueeze(0), (torch.FloatTensor if torch.cuda.is_available() else torch.Tensor)(phase).unsqueeze(0)
        max_mag = np.max(np.abs(m)) # Maximum magnitude value
        real = real / max_mag # Normalize real values
        imag = imag / max_mag # Normalize imaginary values
        #print(f"Applying normalized complex spectrogram: {real.shape}, {imag.shape}")
        return (torch.FloatTensor if torch.cuda.is_available() else torch.Tensor)(real).unsqueeze(0), (torch.FloatTensor if torch.cuda.is_available() else torch.Tensor)(imag).unsqueeze(0)

'''
    def transform(self, audio):
        #m = numpy.abs(librosa.stft(audio/numpy.abs(audio).max(), n_fft= 1024, hop_length=256))[:-1,:]
        m = numpy.abs(librosa.stft(audio, n_fft= 1024, hop_length=256))[:-1,:]
        #print(f"shape of spec: {m.shape}")
        m = numpy.log10(m + self._eps)
        m = (((m - m.min())/(m.max() - m.min()) * 2) - 1)
        #return (torch.FloatTensor if torch.cuda.is_available() else torch.Tensor)(m * 0.8).unsqueeze(0)
        return (torch.FloatTensor if torch.cuda.is_available() else torch.Tensor)(m).unsqueeze(0)


    def inverse(self, spec, maxtractor):
        #print(f"Shape of spec: {spec.shape}")
        s = spec.cpu().detach().numpy()
        #s_ori = numpy.exp((((s + 1) * 0.5) * 19.5) - 17.5) - self._eps # Empirical (average) min and max over test set
        spec_224 = F.interpolate(spec, size=(224,224), mode='bilinear', align_corners=False)
        s_factor = maxtractor.forward(spec_224) # Denormalize the generated spectrogram.
        #print(f"Shape of s_factor: {s_factor.shape}")
        s_max = s_factor[:, 0].view(-1, 1, 1, 1)  # Shape [batch, 1, 1, 1]
        s_min = s_factor[:, 1].view(-1, 1, 1, 1)  # Shape [batch, 1, 1, 1]

        # Expand s_max and s_min to match the shape of spec
        s_max = s_max.expand(-1, 1, 512, 512)  # Shape [batch, 1, 512, 512]
        s_min = s_min.expand(-1, 1, 512, 512)  # Shape [batch, 1, 512, 512]
        
        spec_denorm = numpy.power(10, (s + 1) * (s_max - s_min) / 2 + s_min) - self._eps
        # print(f"Max value of spec: {torch.max(spec)}")
        # print("Max value: ", s_max)
        # print(f"Max value of spec_ori: {numpy.max(s_ori)}")
        # print(f"Max value of spec_denorm: {numpy.max(spec_denorm)}")

        rp = numpy.random.uniform(-numpy.pi, numpy.pi, s.shape)
        f = spec_denorm * (numpy.cos(rp) + (1.j * numpy.sin(rp)))
        print(f"Shape of f: {f.shape}")
        #y = librosa.istft(f) # Reconstruct audio
        # return y
        return librosa.istft(f)


    def inverse(self, spec, maxtractor):
        s = spec  # Keep on GPU
        #print(f"Shape of s: {s.shape}")
        spec_224 = F.interpolate(spec, size=(224,224), mode='bilinear', align_corners=False)
        
        s_factor = maxtractor.forward(spec_224)  # Assume this is a PyTorch tensor on the GPU
        s_max = s_factor[:, 0].view(-1, 1, 1, 1)  # Shape [batch, 1, 1, 1]
        s_min = s_factor[:, 1].view(-1, 1, 1, 1)  # Shape [batch, 1, 1, 1]

        # Expand s_max and s_min to match the shape of spec
        s_max = s_max.expand(-1, 1, 512, 512)  # Shape [batch, 1, 512, 512]
        s_min = s_min.expand(-1, 1, 512, 512)  # Shape [batch, 1, 512, 512]
        #print(f"Shape of s_max: {s_max.shape} and Shape of s_min: {s_min.shape}")
        
        # Denormalize the spectrogram using PyTorch operations
        spec_denorm = torch.pow(10, (s + 1) * (s_max - s_min) / 2 + s_min) - self._eps

        # Generate random phase on GPU
        rp = torch.rand_like(spec) * (2 * torch.pi) - torch.pi
        f = spec_denorm * (torch.cos(rp) + (1.j * torch.sin(rp)))
        f = numpy.complex64(f.detach().cpu())

        # Perform ISTFT with torch.istft (if using PyTorch for ISTFT)
        #return torch.istft(f, n_fft=1024, hop_length=256)  # Adjust n_fft and hop_length as needed
        return librosa.istft(f)
'''