import argparse
import os
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
import matplotlib
import matplotlib.pyplot as plt
from models.networks import Generator
from PIL import Image
import torch.nn.functional as F
matplotlib.use('Agg')


def get_args():
    parser = argparse.ArgumentParser(description='VISAGE inference')
    parser.add_argument('--model_path', '-gp', type=str, help='Path to generator for loading', required=True)
    parser.add_argument('--data_path', '-d', type=str, help='Path to inference data', required=True)
    parser.add_argument('--output_path', '-o', type=str, help='Where to save the trained images', required=True)
    parser.add_argument('--input_age', '-ia', type=int, help='Age of the person in the input images', required=True)
    parser.add_argument('--output_age', '-oa', type=int, help='Age of the person in the input images', required=True)
    parser.add_argument('--resize', '-r', type=int, help='Resize output image to certain value')

    return parser.parse_args()


def save_image(input_image, delta, input_age, target_age, filename):
    # Function to plot side by side images of input, delta and output images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(normalise_permute_image(input_image))
    axs[0].set_title(f'Input - age = {input_age}')
    axs[0].axis('off')
    axs[1].imshow(normalise_permute_image(delta))
    axs[1].set_title('Delta')
    axs[1].axis('off')
    axs[2].imshow(normalise_permute_image((input_image.to('cpu')+delta)))
    axs[2].set_title(f'Output - age = {target_age}')
    axs[2].axis('off')

    plt.savefig(filename)
    plt.close(fig)


def normalise_permute_image(input_image):
    # Function to normalise [0,1] and permute the image for plotting

    clamped_image = torch.clamp(input_image, -1, 1)
    return clamped_image.permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5


def age_to_tensor(age):
    size = 512
    age = age/100
    tensor = torch.full((1, size, size), age).unsqueeze(0)
    return tensor


def upsample(tensor, size):
    output_tensor = F.interpolate(tensor, size=(size, size), mode='bilinear', align_corners=False)
    return output_tensor


def infer():

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # results save path
    data_path = args.data_path
    output_path = args.output_path
    input_age = args.input_age
    output_age = args.output_age
    if not os.path.isdir(data_path) or not os.path.exists(data_path):
        print('Data path does not exist!')
        return 0
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # Load generator
    generator = Generator().to(device)
    if not args.model_path:
        print('No model path provided. Exiting...')
        return 0
    saved_state_dict = torch.load(args.model_path, map_location=device)
    generator.load_state_dict(saved_state_dict)

    transform = transforms.Compose([
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    file_list = sorted(os.listdir(data_path) if os.path.isdir(data_path) else [data_path])
    for filename in tqdm(file_list, desc="Processing files"):
        # Load and process images
        filepath = os.path.join(data_path, filename)
        input_img = transform(transforms.ToTensor()(Image.open(filepath).convert('RGB')).to(device))

        input_img_size = input_img.size()[2]

        input_img_resized = transform(transforms.ToTensor()(Image.open(filepath).convert('RGB').resize((512, 512))).to(device))
        input_age_tensor = age_to_tensor(input_age)
        output_age_tensor = age_to_tensor(output_age)

        # Model input is the input image, input age and target age
        model_input = torch.cat((input_img_resized.unsqueeze(0), input_age_tensor, output_age_tensor), dim=1).to(device)

        # Generate delta
        with torch.no_grad():
            generator.eval()
            delta = generator(model_input)
            delta = upsample(delta, size=input_img_size)

            output_image = input_img + delta
            image = normalise_permute_image(output_image.squeeze(0))

        # plt.imshow(image)

        # Resize image
        if args.resize:
            size = (args.resize, args.resize)
            image = Image.fromarray((255 * image).round().astype('uint8')).resize(size, Image.ANTIALIAS)
        else:
            image = Image.fromarray((255 * image).round().astype('uint8'))

        # Could also use save_image function to plot side by side images

        # Save the output image
        outfile = os.path.join(output_path, filename)
        # save_image(input_img.squeeze(0), delta.squeeze(0), input_age, output_age, outfile)

        image.save(outfile)
        image.close()

if __name__ == '__main__':
    infer()
