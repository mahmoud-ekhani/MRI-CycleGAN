import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_diffusion(
    model,
    trainer,
    dataloader,
    num_epochs,
    device,
    save_dir,
    save_interval=1000
):
    """
    Train the diffusion model using the updated DDPMTrainer, which handles:
      - Optimizer & mixed precision steps
      - Forward/backward within train_one_batch
      - Sampler for generation
    """

    # Make sure 'model' inside 'trainer' is used, so we typically don't
    # separately reference model here except for saving states or setting eval mode.
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    
    for epoch in range(num_epochs):
        # Put trainer model in train mode
        trainer.model.train()
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            t1 = batch['T1'].to(device)  # shape [B, 1, H, W]
            t2 = batch['T2'].to(device)
            
            # T1 -> T2
            loss_t1_t2 = trainer.train_one_batch(
                x_0=t2,            # Target domain is T2
                condition=t1,      # Condition is T1
                context=t1         # Cross-attention sees T1
            )

            # T2 -> T1
            loss_t2_t1 = trainer.train_one_batch(
                x_0=t1,
                condition=t2,
                context=t2         # Cross-attention sees T2
            )
            
            # Checkpoint & sampling
            if global_step % save_interval == 0:
                # Save states
                trainer.model.eval()
                torch.save({
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                }, save_dir / f'model_{global_step}.pt')
                
                # Generate samples
                with torch.no_grad():
                    # We'll sample T1->T2 from the first 4 T1 images
                    t2_sample = trainer.sample(
                        condition=t1[:4],
                        shape=t1[:4].shape   # shape = [4, 1, H, W]
                    )
                    # We'll sample T2->T1 from the first 4 T2 images
                    t1_sample = trainer.sample(
                        condition=t2[:4],
                        shape=t2[:4].shape
                    )
                    
                    # Visualize
                    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                    for i in range(4):
                        # T1->T2
                        axes[0,i].imshow(t2_sample[i,0].cpu().numpy(), cmap='gray')
                        axes[0,i].set_title(f'Generated T2 {i+1}')
                        # T2->T1
                        axes[1,i].imshow(t1_sample[i,0].cpu().numpy(), cmap='gray')
                        axes[1,i].set_title(f'Generated T1 {i+1}')
                    
                    plt.tight_layout()
                    plt.savefig(save_dir / f'samples_{global_step}.png')
                    plt.close()
                
                trainer.model.train()
            
            global_step += 1


# Example usage
if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Suppose you have a UNet that expects 2 channels (x + condition)
    from diffusion_model import UNet
    from diffusion_trainer import DDPMTrainer  # The updated trainer
    from diffusion_dataset import MRIDiffusionDataset
    
    # Build model & trainer
    model = UNet(in_channels=2)
    trainer = DDPMTrainer(model=model, device=device, lr=1e-4)
    
    # Build dataset & loader
    dataset = MRIDiffusionDataset(t1_dir='path/to/t1', t2_dir='path/to/t2')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # Train
    train_diffusion(
        model=model,       # needed only for references like model.state_dict() in checkpoint
        trainer=trainer,
        dataloader=dataloader,
        num_epochs=100,
        device=device,
        save_dir='diffusion_checkpoints'
    )
