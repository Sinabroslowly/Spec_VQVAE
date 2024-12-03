import os
import soundfile
import torch
from torch.utils.data import Dataset
from .stft import STFT

def make_dataset(dir):
    audio_files = []
    assert os.path.isdir(dir), "%s is not a valid directory." % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith((".wav", ".WAV", ".aif", ".aiff", ".AIF", ".AIFF")):
                path = os.path.join(root, fname)
                audio_files.append(path)

    return audio_files

class AudioDataset(Dataset):
    def __init__(self, dataroot, phase="train", spec="stft"):
        self.phase = phase
        self.root = os.path.join(dataroot, phase)
        self.stft = STFT()
        
        # Audio files only
        self.audio_paths = sorted(make_dataset(self.root))

    def __getitem__(self, index):
        if index >= len(self):
            return None

        # Load audio file and convert to spectrogram
        audio_path = self.audio_paths[index]
        audio, _ = soundfile.read(audio_path)
        audio_spec = self.stft.transform(audio)

        return audio_spec, audio_path

    def __len__(self):
        return len(self.audio_paths)

    def name(self):
        return "AudioDataset"
    


def main(phase):
    # Specify the dataset root directory and phase (e.g., "train", "val", "test")
    dataroot = "./datasets"

    # Initialize the dataset
    dataset = AudioDataset(dataroot=dataroot, phase=phase)

    # Check the dataset length
    print(f"Total samples in {phase} phase: {len(dataset)}")

    # Load the first sample to test
    if len(dataset) > 0:
        audio_spec, audio_path = dataset[0]
        print(f"Loaded audio from: {audio_path}")
        print(f"Spectrogram Normalized real shape: {audio_spec[0].shape}")
        print(f"Spectrogram normalized imag shape: {audio_spec[1].shape}")
    else:
        print("No audio files found in the specified directory.")

if __name__ == "__main__":
    phase = "val"
    main(phase)
