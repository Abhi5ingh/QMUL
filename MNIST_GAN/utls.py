import matplotlib.pyplot as plt
import torch

def create_noise(num, dim):
    return torch.randn(num, dim)

def show_result(G_net, z_, num_epoch, save=False, path='result.png'):
    fig, axs = plt.subplots(5, 5, figsize=(5, 5))
    fig.suptitle(f'Epoch #{num_epoch}')
    for ax, image in zip(axs.flat, G_net(z_).cpu().detach().numpy()):
        ax.imshow(image.reshape((28, 28)), cmap='gray')

    if save:
        fig.savefig(path)
        plt.close()
