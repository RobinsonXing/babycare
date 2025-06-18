import re
import matplotlib.pyplot as plt

def visualize_gan_losses(log_file_path):
    epochs = []
    d_losses = []
    g_losses = []

    pattern = re.compile(
        r"\[Epoch\s+(\d+)/\d+\]\s+Discriminator_Loss:\s+([-+]?[0-9]*\.?[0-9]+),\s+Generator_Loss:\s+([-+]?[0-9]*\.?[0-9]+)"
    )

    with open(log_file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                d_loss = float(match.group(2))
                g_loss = float(match.group(3))
                epochs.append(epoch)
                d_losses.append(d_loss)
                g_losses.append(g_loss)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, d_losses, label="Discriminator Loss", color='red')
    plt.plot(epochs, g_losses, label="Generator Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("WGAN-GP Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()