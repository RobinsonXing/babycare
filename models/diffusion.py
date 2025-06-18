import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianDiffusion:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps

        # diffusion schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def to(self, device):
        """Move all internal tensors to the given device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, label):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, label)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample_loop(self, shape, label):
        device = next(self.model.parameters()).device
        B = shape[0]
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = self.model(x, label)
            alpha = self.alphas[t]
            alpha_bar = self.alphas_cumprod[t]
            beta = self.betas[t]

            x = (1 / torch.sqrt(alpha)) * (
                x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise
            )
            if t > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(beta) * noise
        return x


class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, seq_len, num_layers=2):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, hidden_dim)
        self.lstm = nn.LSTM(input_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, label):
        B, T, _ = x.shape
        label_embedding = self.label_emb(label).unsqueeze(1).repeat(1, T, 1)
        x_cond = torch.cat([x, label_embedding], dim=-1)
        h, _ = self.lstm(x_cond)
        return self.output(h)