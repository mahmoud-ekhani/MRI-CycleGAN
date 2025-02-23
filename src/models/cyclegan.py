import torch
import torch.nn as nn
import itertools
from .networks import Generator, Discriminator

class CycleGAN:
    def __init__(self, device="cuda"):
        self.device = device

        # Generators
        self.G_XtoY = Generator(1, 1).to(device)
        self.G_YtoX = Generator(1, 1).to(device)

        # Discriminators
        self.D_X = Discriminator(1).to(device)
        self.D_Y = Discriminator(1).to(device)

        # Optimizers
        self.opt_G = torch.optim.Adam(
            itertools.chain(self.G_XtoY.parameters(), self.G_YtoX.parameters()),
            lr=2e-4, betas=(0.5, 0.999)
        )
        self.opt_D_X = torch.optim.Adam(self.D_X.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_D_Y = torch.optim.Adam(self.D_Y.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Losses
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # Loss weights
        self.lambda_cycle = 10.0
        self.lambda_id = 0.5

        # Learning rate schedulers
        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_G, T_max=200, eta_min=1e-5
        )
        self.scheduler_D_X = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_D_X, T_max=200, eta_min=1e-5
        )
        self.scheduler_D_Y = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_D_Y, T_max=200, eta_min=1e-5
        )

        # Training history
        self.history = {
            'G_losses': [], 'D_X_losses': [], 'D_Y_losses': [],
            'cycle_losses': [], 'identity_losses': []
        }

    def set_input(self, x, y):
        self.real_X = x.to(self.device)
        self.real_Y = y.to(self.device)

    def forward(self):
        # X->Y
        self.fake_Y = self.G_XtoY(self.real_X)
        self.rec_X = self.G_YtoX(self.fake_Y)
        # Y->X
        self.fake_X = self.G_YtoX(self.real_Y)
        self.rec_Y = self.G_XtoY(self.fake_X)

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
        
        pred_fake = netD(fake.detach())
        loss_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        
        return (loss_real + loss_fake) * 0.5

    def backward_D_X(self):
        self.loss_D_X = self.backward_D_basic(self.D_X, self.real_X, self.fake_X)
        self.loss_D_X.backward()

    def backward_D_Y(self):
        self.loss_D_Y = self.backward_D_basic(self.D_Y, self.real_Y, self.fake_Y)
        self.loss_D_Y.backward()

    def backward_G(self):
        # Adversarial loss
        pred_fake_Y = self.D_Y(self.fake_Y)
        self.loss_G_XtoY = self.criterion_GAN(pred_fake_Y, torch.ones_like(pred_fake_Y))
        
        pred_fake_X = self.D_X(self.fake_X)
        self.loss_G_YtoX = self.criterion_GAN(pred_fake_X, torch.ones_like(pred_fake_X))

        # Cycle consistency loss
        self.loss_cycle_X = self.criterion_cycle(self.rec_X, self.real_X) * self.lambda_cycle
        self.loss_cycle_Y = self.criterion_cycle(self.rec_Y, self.real_Y) * self.lambda_cycle

        # Identity loss
        idt_X = self.G_YtoX(self.real_X)
        idt_Y = self.G_XtoY(self.real_Y)
        self.loss_idt_X = self.criterion_identity(idt_X, self.real_X) * self.lambda_id
        self.loss_idt_Y = self.criterion_identity(idt_Y, self.real_Y) * self.lambda_id

        # Combined generator loss
        self.loss_G = (self.loss_G_XtoY + self.loss_G_YtoX +
                      self.loss_cycle_X + self.loss_cycle_Y +
                      self.loss_idt_X + self.loss_idt_Y)
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        
        # Update generators
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

        # Update discriminator X
        self.opt_D_X.zero_grad()
        self.backward_D_X()
        self.opt_D_X.step()

        # Update discriminator Y
        self.opt_D_Y.zero_grad()
        self.backward_D_Y()
        self.opt_D_Y.step()

        # Update learning rates
        self.scheduler_G.step()
        self.scheduler_D_X.step()
        self.scheduler_D_Y.step()

        # Update history
        self.history['G_losses'].append(self.loss_G.item())
        self.history['D_X_losses'].append(self.loss_D_X.item())
        self.history['D_Y_losses'].append(self.loss_D_Y.item())
        self.history['cycle_losses'].append(self.loss_cycle_X.item() + self.loss_cycle_Y.item())
        self.history['identity_losses'].append(self.loss_idt_X.item() + self.loss_idt_Y.item()) 