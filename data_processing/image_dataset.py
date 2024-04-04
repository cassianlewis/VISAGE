from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
import os
from typing import Type

from data_processing.augmentations import CustomTransform


class ImageDataset(Dataset):
    def __init__(self, data_dir: str, transform: Type = None, augmentation: bool = False):
        self.data = []
        folders = sorted(os.listdir(data_dir))
        file_list = sorted(os.listdir(f"{data_dir}/{folders[0]}"))

        self.ages = [int(f) for f in folders]
        self.num_ages = len(folders)
        self.num_faces = len(file_list)

        for file in file_list:
            img_paths = [f"{data_dir}/{folder}/{file}" for folder in folders]
            self.data.append(img_paths)

        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data) * self.num_ages**2

    def __getitem__(self, index):
        # Calculate face index and input/output age indices.
        face_idx = index // self.num_ages**2
        input_age_idx = (index % self.num_ages**2) // self.num_ages
        output_age_idx = index % self.num_ages

        # Specify input and output paths using indices
        input_path = self.data[face_idx][input_age_idx]
        output_path = self.data[face_idx][output_age_idx]
        input_age = self.ages[input_age_idx]
        output_age = self.ages[output_age_idx]

        # Load images using
        input_img = Image.open(input_path).convert('RGB')
        output_img = Image.open(output_path).convert('RGB')

        # Transform images with probability = 0.5
        if self.augmentation:
            p = random.uniform(0, 1)
            if p > 0.5:
                # Define hue, colour values etc for augmentation
                a, b, c, d = random.uniform(0.93, 1.07), random.uniform(0.93, 1.07), random.uniform(0.93, 1.07), random.uniform(-0.04,0.04)
                rotation = random.uniform(-10, 10)
                # Add gaussian blur to imitate motion effects
                gauss = random.uniform(0.1, 4)
                kernel_size = (11, 11)

                # Define crop boundaries
                img_width, img_height = input_img.size
                # Adjust crop size to be 5/6 of image size
                crop_size = int(5 / 6 * min(img_width, img_height))
                left = random.randint(0, img_width - crop_size)
                top = random.randint(0, img_height - crop_size)

                # Define and call CustomTransform class
                custom_transform = CustomTransform(a, b, c, d, rotation, kernel_size, gauss, left, top)
                input_img = custom_transform(input_img)
                output_img = custom_transform(output_img)

        # Normalise tensors in range [-1,1]
        input_img_raw = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(input_img))
        output_img_raw = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(output_img))

        # Resize tensor to 512x512
        input_img_resized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(input_img.resize((512,512))))
        output_img_resized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(output_img.resize((512,512))))

        return input_img_resized, output_img_resized, input_age, output_age, input_img_raw, output_img_raw

    def upsample(self, img):
        output_image = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
        return output_image




