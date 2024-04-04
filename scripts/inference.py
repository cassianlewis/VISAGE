
import argparse
import os
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.models import Generator
from PIL import Image
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(description='VISAGE inference')
    parser.add_argument('--model_path', '-gp', type=str, help='Path to generator for loading', required=True)
    parser.add_argument('--data_path', '-d', type=str, help='Path to training data', required=True)
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
    axs[2].imshow(normalise_permute_image((input_image.to('cuda')+delta)))
    axs[2].set_title(f'Output - age = {target_age}')
    axs[2].axis('off')

    plt.savefig(filename)
    plt.close(fig)


def normalise_permute_image(input_image):
    return input_image.permute(1, 2, 0).cpu().numpy()*0.5+0.5


def age_to_tensor(age):
    size = 512
    age=age/100
    tensor = torch.full((1, size, size), age).unsqueeze(0)
    return tensor.to('cuda')


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
    if not os.path.isdir(data_path):
        print('Data path does not exist!')
        return 0
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # Load generator
    generator = Generator().to(device)
    if not args.model_path:
        print('No model path provided. Exiting...')
        return 0
    saved_state_dict = torch.load(args.model_path)
    generator.load_state_dict(saved_state_dict)

    transform = transforms.Compose([
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    file_list = os.listdir(data_path)
    for filename in tqdm(file_list, desc="Processing files"):
        # Load and process images
        filepath = os.path.join(data_path, filename)
        input_img = transform(transforms.ToTensor()(Image.open(filepath).convert('RGB'))).to(device)
        input_img_size = input_img.size()[2]
        input_img_resized = transform(transforms.ToTensor()(Image.open(filepath).convert('RGB').resize((512,512))).to(device))
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

        # Resize image
        if args.resize:
            size = (args.resize, args.resize)
            image = Image.fromarray((255 * image).round().astype('uint8')).resize(size, Image.ANTIALIAS)
        else:
            image = Image.fromarray((255 * image).round().astype('uint8'))

        # Could also use save_image function to plot side by side images
        image.save(os.path.join(output_path, filename))
        image.close()

if __name__ == '__main__':
    infer()

# ffmpeg -framerate 25 -pattern_type glob -i 'blair_frames/*.png' -c:v libx264 -vf fps=25 -pix_fmt yuv420p blair_unedited.mp4
# ffmpeg -framerate 25 -pattern_type glob -i 'theresa_frames_cropped_edited/*.jpg' -c:v libx264 -vf fps=25 -pix_fmt yuv420p theresa_edited_old.mp4
# ffmpeg -i theresa_unedited.mp4 -i theresa_edited_young.mp4 -i theresa_edited_old.mp4 -filter_complex "[0:v]scale=1024:1024,drawtext=text='Age=60':fontsize=24:x=10:y=10:fontcolor=white[video1]; [1:v]scale=1024:1024,drawtext=text='Age=30':fontsize=24:x=10:y=10:fontcolor=white[video2]; [2:v]scale=1024:1024,drawtext=text='Age=80':fontsize=24:x=10:y=10:fontcolor=white[video3]; [video1][video2][video3]hstack=inputs=3" theresa_combined.mp4


# ffmpeg -framerate 25 -pattern_type glob -i 'blair_frames/*.png' -c:v libx264 -vf fps=25 -pix_fmt yuv420p blair_unedited.mp4
# ffmpeg -framerate 25 -pattern_type glob -i 'blair_frames_range/*.png' -c:v libx264 -vf fps=25 -pix_fmt yuv420p blair_range.mp4
# ffmpeg -framerate 25 -pattern_type glob -i 'blair_frames_80/*.png' -c:v libx264 -vf fps=25 -pix_fmt yuv420p blair_edited_80.mp4

# ffmpeg -i blair_unedited.mp4 -i blair_edited_30.mp4 -i blair_edited_80.mp4 -filter_complex "[0:v]scale=1024:1024,drawtext=text='Age=60':fontsize=24:x=10:y=10:fontcolor=white[video1]; [1:v]scale=1024:1024,drawtext=text='Age=30':fontsize=24:x=10:y=10:fontcolor=white[video2]; [2:v]scale=1024:1024,drawtext=text='Age=80':fontsize=24:x=10:y=10:fontcolor=white[video3]; [video1][video2][video3]hstack=inputs=3" theresa_combined.mp4
# ffmpeg -i blair_unedited.mp4 -i blair_edited_30.mp4 -i blair_edited_80.mp4 -filter_complex "[0:v]drawtext=text='Age=60':fontsize=24:x=10:y=10:fontcolor=white[video1]; [1:v]drawtext=text='Age=30':fontsize=24:x=10:y=10:fontcolor=white[video2]; [2:v]drawtext=text='Age=80':fontsize=24:x=10:y=10:fontcolor=white[video3]; [video1][video2][video3]hstack=inputs=3" blair_combined.mp4

# ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 blair_unedited.mp4

# ffmpeg -i blair.webm -ss 00:00:02 -t 00:00:18 -vn -c:a libmp3lame -q:a 2 blair_audio.mp3
# ffmpeg -i blair_edited_30.mp4 -i blair_audio.mp3 -c:v copy -map 0:v -map 1:a -shortest blair_edited_30_sound.mp4