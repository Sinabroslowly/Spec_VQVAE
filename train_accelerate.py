import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
import pyroomacoustics
from diffusers import VQModel
from accelerate import Accelerator
import argparse
import lpips
import numpy as np

from scripts.dataset import AudioDataset
from scripts.stft import STFT



LR = 1e-5
ADAM_BETA = (0.0, 0.99)
ADAM_EPS = 1e-8
ENC_LR = 2e-4
QUANT_LR = 2e-4
DEC_LR = 2e-4
LAMBDA_1 = 5e+1
LAMBDA_2 = 1e+2
LAMBDA_3 = 1e+1
LAMBDAS = [1, 1e-2, 1, 1] # [Spec Recon, Quantization, T60_error, LPIPS Loss]

def train(model, train_loader, val_loader, optimizer, scheduler, device, stft, lpips_loss, start_epoch, best_val_loss, args, accelerator):
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(args.logs, args.version))
    
    for epoch in range(start_epoch + 1, args.epochs):
        model.train()
        train_loss_total = 0
        train_loss_1 = 0
        train_loss_2 = 0
        train_loss_3 = 0

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train: Epoch {epoch}/{args.epochs}")
        for _ , batch in train_pbar:
            audio_spec, _ = batch
            input_data = audio_spec.to(device)
            output, commitment_loss = model(input_data, return_dict=False)

            # Loss 1: Reconstruction loss MAE
            reconstruction_spec_loss = nn.functional.l1_loss(output, input_data)
            
            # Loss 3: RT60 loss (based on pyroomacoustics)
            y_r = [stft.inverse(s.squeeze()) for s in input_data]
            y_f = [stft.inverse(s.squeeze()) for s in output]

            t60_loss = 1
            try:
                f = lambda x: pyroomacoustics.experimental.rt60.measure_rt60(x, 22050)
                t60_r = [f(y) for y in y_r if len(y)]
                t60_f = [f(y) for y in y_f if len(y)]
                t60_loss = np.mean([((t_b - t_a) / t_a) for t_a, t_b in zip(t60_r, t60_f)])
            except:
                pass

            loss = (
                LAMBDAS[0] * reconstruction_spec_loss +
                LAMBDAS[1] * args.commitment_cost * commitment_loss +
                LAMBDAS[2] * t60_loss
            )

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            train_loss_total += loss.item()
            train_loss_1 += reconstruction_spec_loss.item()
            train_loss_2 += commitment_loss.item()
            train_loss_3 += t60_loss

        # Logging training metrics
        if accelerator.is_main_process:
            writer.add_scalar("Loss/Total_Train", train_loss_total / len(train_loader), epoch)
            writer.add_scalar("Loss/Reconstruction_Train", train_loss_1 / len(train_loader), epoch)
            writer.add_scalar("Loss/Commitment_Train", train_loss_2 / len(train_loader), epoch)
            writer.add_scalar("Loss/T60_Train", train_loss_3 / len(train_loader), epoch)

            print(f"Epoch {epoch}/{args.epochs}, Loss: {train_loss_total / len(train_loader):.4f}")

        # Validation Step
        model.eval()
        val_loss_total = 0
        val_loss_1 = 0
        val_loss_2 = 0
        val_loss_3 = 0
        val_loss_4 = 0

        with torch.no_grad():
            val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation: Epoch {epoch}/{args.epochs}")
            for _, batch in val_pbar:
                audio_spec, _ = batch
                input_data = audio_spec.to(device)
                output, commitment_loss = model(input_data, return_dict=False)

                reconstruction_spec_loss = nn.functional.l1_loss(output, input_data)
                y_r = [stft.inverse(s.squeeze()) for s in input_data]
                y_f = [stft.inverse(s.squeeze()) for s in output]

                t60_loss = 1
                try:
                    f = lambda x: pyroomacoustics.experimental.rt60.measure_rt60(x, 22050)
                    t60_r = [f(y) for y in y_r if len(y)]
                    t60_f = [f(y) for y in y_f if len(y)]
                    t60_loss = np.mean([((t_b - t_a) / t_a) for t_a, t_b in zip(t60_r, t60_f)])
                except:
                    pass

                original_spec_rgb = input_data.repeat(1, 3, 1, 1) # [batch, 3, height, width]
                reconstructed_spec_rgb = output.repeat(1, 3, 1, 1) # [batch, 3, height, width]
                lpips_spec_loss = lpips_loss(original_spec_rgb, reconstructed_spec_rgb).mean()

                loss = (
                    LAMBDAS[0] * reconstruction_spec_loss +
                    LAMBDAS[1] * args.commitment_cost * commitment_loss +
                    LAMBDAS[2] * t60_loss +
                    LAMBDAS[3] * lpips_spec_loss
                )

                val_loss_total += loss.item()
                val_loss_1 += reconstruction_spec_loss.item()
                val_loss_2 += commitment_loss.item()
                val_loss_3 += t60_loss
                val_loss_4 += lpips_spec_loss

        # Logging validation metrics
        if accelerator.is_main_process:
            writer.add_scalar("Validation/Total_Validation", val_loss_total / len(val_loader), epoch)
            writer.add_scalar("Validation/Reconstruction_Validation", val_loss_1 / len(val_loader), epoch)
            writer.add_scalar("Validation/Commitment_Validation", val_loss_2 / len(val_loader), epoch)
            writer.add_scalar("Validation/T60_Validation", val_loss_3 / len(val_loader), epoch)
            writer.add_image("Spectrogram/Ground_Truth", make_grid(input_data.cpu(), normalize=True), epoch)
            writer.add_image("Spectrogram/Generated", make_grid(output.cpu(), normalize=True), epoch)
        
            print(f"Validation Loss: {val_loss_total / len(val_loader):.4f}")

        # Save checkpoints
        if accelerator.is_main_process:
            if (epoch + 1) % (args.epochs // 5) == 0 or val_loss_total < best_val_loss:
                checkpoint_path = os.path.join(
                    args.checkpoints_dir,
                    args.version,
                    f"{'best_val_checkpoint' if val_loss_total < best_val_loss else f'epoch_{epoch + 1}_checkpoint'}.pth"
                )
                torch.save({
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_loss": min(val_loss_total, best_val_loss),
                    }, checkpoint_path)
                
                if val_loss_total < best_val_loss:
                    best_val_loss = val_loss_total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Path to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size")
    parser.add_argument("--dataset", type=str, default="./datasets", help="Dataset path")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs to train the model.")
    parser.add_argument("--from_pretrained", type=str, default=None, help="The checkpoint name for pretraining")
    parser.add_argument("--version", type=str, default="trial_02", help="The version of VQ-VAE training")
    parser.add_argument("--embedding_dim", type=int, default=3, help="Embedding dimension for VQ-VAE.")
    parser.add_argument("--num_embeddings", type=int, default=128, help="Number of embeddings for VQ-VAE.")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="Commitment cost for VQ-VAE.")
    parser.add_argument("--logs", type=str, default="./logs", help="Logging directory for Tensorboard.")
    parser.add_argument("--function_test", type=bool, default=False, help="Flag to determine model function and memory allocation test.")
    args = parser.parse_args()

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoints_dir, args.version), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Accelerator setup
    accelerator = Accelerator()
    device = accelerator.device

    # Model setup
    model = VQModel(
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(32, 64, 128, 256),
        layers_per_block=2,
        act_fn="silu",
        sample_size=512,
        latent_channels=8,
        num_vq_embeddings=256,
        scaling_factor=0.18215,
    )
    start_epoch = 0
    best_val_loss = float("inf")
    if args.from_pretrained:
        checkpoint = torch.load(os.path.join(args.checkpoints_dir, args.version, args.from_pretrained), map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        #print(f"Loaded checkpoint from: {args.from_pretrained}")


    train_dataset = AudioDataset(dataroot="./datasets", phase="train", spec="stft")
    val_dataset = AudioDataset(dataroot="./datasets", phase="val", spec="stft")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    lpips_loss = lpips.LPIPS(net='vgg').to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=ADAM_BETA, eps=ADAM_EPS)

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model,
        optimizer,
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100),
        train_loader,
        val_loader
    )
    train(model, train_loader, val_loader, optimizer, scheduler, device, STFT(), lpips_loss, start_epoch, best_val_loss, args=args, accelerator=accelerator)

if __name__ == "__main__":
    main()
