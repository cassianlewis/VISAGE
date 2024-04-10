import argparse
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from utils.coach import FRANTrainer
from models.networks import Generator, Discriminator
from data_processing.image_dataset import ImageDataset
import torch.nn.functional as F
matplotlib.use('Agg')


def get_args():
    parser = argparse.ArgumentParser(description='Train VISAGE for re-aging')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='Batch size')
    parser.add_argument('--eval_steps', '-es', type=int, default=1000, help='Number of steps at which the model is evaluated')
    parser.add_argument('--model_path_gen', '-gp', type=str, help='Path to generator for loading')
    parser.add_argument('--model_path_disc', '-dp', type=str, help='Path to discriminator for loading')
    parser.add_argument('--data_path', '-d', type=str, help='Path to training data', required=True)
    parser.add_argument('--exp_dir', '-t', type=str, help='Path to intermediate training images/losses/models', default='experiments')
    parser.add_argument('--version', '-v', type=str, help='Model version number eg V5', default='V1')
    parser.add_argument('--augmentation', '-a', type=bool, help='Whether or not to augment the training data', default=False)
    return parser.parse_args()


def save_image(input_image, delta, target_image, input_age, target_age, images_path, index, epoch, device):
    # Function to plot side by side input, delta, output and ground truth images

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(normalise_permute_image(input_image))
    axs[0].set_title(f'Input - age = {input_age}')
    axs[0].axis('off')
    axs[1].imshow(normalise_permute_image(delta))
    axs[1].set_title('Delta')
    axs[1].axis('off')
    axs[2].imshow(normalise_permute_image((input_image.to(device)+delta)))
    axs[2].set_title(f'Output - age = {target_age}')
    axs[2].axis('off')
    axs[3].imshow(normalise_permute_image(target_image))
    axs[3].set_title(f'Ground truth - age = {target_age}')
    axs[3].axis('off')

    filename = os.path.join(images_path, f'{epoch}_{index}.jpeg')
    plt.savefig(filename)
    plt.close(fig)


def plot_rolling_average(train_hist, window_size, losses, filename):
    # Function to plot the rolling average of the losses
    rolling_losses = {}
    for loss, key in losses.items():
        rolling_losses[loss] = np.convolve(train_hist[key], np.ones(window_size)/window_size, mode='valid')

    # Create plot
    plt.figure(figsize=(10, 5), dpi=300)
    for loss, color in zip(losses.keys(), ['green', 'blue', 'red']):
        plt.plot(rolling_losses[loss], color=color, label=loss, linewidth=0.8)
    plt.legend()
    plt.xlabel('Training step')
    plt.ylabel('Loss')
    plt.savefig(fname=filename)
    plt.close()


def normalise_permute_image(input_image):
    # Function to normalise [0,1] and permute the image for plotting
    return (input_image.permute(1, 2, 0).cpu().numpy() - input_image.min().item()) / (input_image.max().item() - input_image.min().item())


def upsample(img):
    output_image = F.interpolate(img.unsqueeze(0), scale_factor=2, mode='bilinear', align_corners=False)
    return output_image.squeeze(0)


def train():
    args = get_args()
    epochs = args.epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    # Create saving directories
    experiment_path = os.path.join(args.exp_dir, args.version)
    os.makedirs(experiment_path, exist_ok=True)

    model_output_path = os.path.join(experiment_path, 'models')
    os.makedirs(model_output_path, exist_ok=True)

    images_path = os.path.join(experiment_path, 'images')
    os.makedirs(images_path, exist_ok=True)

    losses_path = os.path.join(experiment_path, 'losses')
    os.makedirs(losses_path, exist_ok=True)

    # Define the models and load weights if needed
    generator = Generator()
    discriminator = Discriminator()
    if args.model_path_gen:
        saved_state_dict = torch.load(args.model_path_gen, map_location=torch.device('cpu'))
        generator.load_state_dict(saved_state_dict)

    if args.model_path_disc:
        saved_state_dict = torch.load(args.model_path_disc)
        discriminator.load_state_dict(saved_state_dict)

    # Create the dataset, dataloader and coach
    dataset = ImageDataset(args.data_path, augmentation=args.augmentation)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    age_list = dataset.ages
    coach = FRANTrainer(generator, discriminator, device=device, batch_size=args.batch_size, age_list=age_list)

    print(f'Batch size  = {args.batch_size}')
    print(f'Total number of epochs = {args.epochs}')
    print(f'Training steps per epoch = {int(len(dataset)/args.batch_size)}')

    # Define dictionary for logging losses
    train_hist = {'G_losses': [], 'D_losses': [], 'G_adv': [], 'G_L1': [], 'G_perc': [],
                  'D_real': [], 'D_fake': [], 'D_fake_ages': []}

    # Train the model
    for epoch in range(epochs):
        for i, (input_images, output_images, input_ages, target_ages, input_images_raw, output_images_raw) in enumerate(
                tqdm(dataloader, desc=f'Epoch {epoch}/{epochs}', unit='batch')):

            # Take training step and update loss dictionary
            generator.train()
            discriminator.train()
            loss_dict = coach.training_step(input_images, output_images, input_ages, target_ages)

            train_hist['G_losses'].append(loss_dict['G_total'])
            train_hist['D_losses'].append(loss_dict['D_total'])
            train_hist['G_adv'].append(loss_dict['G_adv'])
            train_hist['G_L1'].append(loss_dict['G_L1'])
            train_hist['G_perc'].append(loss_dict['G_perc'])
            train_hist['D_real'].append(loss_dict['D_real'])
            train_hist['D_fake'].append(loss_dict['D_fake'])
            train_hist['D_fake_ages'].append(loss_dict['D_fake_ages'])

            print('Step = {0} | gen_loss = {1:.3f} | dis_loss = {2:.3f}'.format(i, loss_dict['G_total'],
                                                                                loss_dict['D_total']))
            # Generate and plot images to assess model performance
            if (i+1) % args.eval_steps == 0:
                deltas = coach.infer(input_images, input_ages, target_ages)
                for x in range(len(input_images)):
                    save_image(input_images_raw[x], upsample(deltas[x]), output_images_raw[x], input_ages[x],
                               target_ages[x], images_path, i+x, epoch, device)

        losses_G = {
            'Adversarial': 'G_adv',
            'L1': 'G_L1',
            'Perceptual': 'G_perc'}

        losses_D = {
            'Real': 'D_real',
            'Fake': 'D_fake',
            'Fake ages': 'D_fake_ages'}

        # Plot losses and save models every epoch
        plot_rolling_average(train_hist, window_size=100, losses=losses_G, filename=f'{losses_path}/gen_losses.png')
        plot_rolling_average(train_hist, window_size=100, losses=losses_D, filename=f'{losses_path}/dis_losses.png')

        coach.save_generator(model_output_path+f'/generator_{args.version}_epoch_{epoch+1}')
        coach.save_discriminator(model_output_path + f'/discriminator_{args.version}_epoch_{epoch+1}')

if __name__ == '__main__':
    train()

