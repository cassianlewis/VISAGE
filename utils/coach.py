import torch
import torch.optim as optim
import torch.nn as nn
from lpips import LPIPS
import random
from typing import Union


class FRANTrainer:
    def __init__(self, generator: nn.Module, discriminator: nn.Module, age_list: list, device: str = "cuda", batch_size: int = 8):
       """
       Initializes the FRANTrainer class.

       Args:
           generator (nn.Module): The generator model.
           discriminator (nn.Module): The discriminator model.
           age_list (list): A list of possible ages.
           device (str, optional): The device to use for computations. Defaults to "cuda".
           batch_size (int, optional): The batch size. Defaults to 8.
       """
       self.device = torch.device(device)
       self.batch_size = batch_size
       self.age_list = age_list

       self.generator = generator.to(self.device)
       self.discriminator = discriminator.to(self.device)

       self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
       self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

       self.lpips = LPIPS(net="vgg").to(self.device)

       self.lambda_L1 = 1
       self.lambda_perceptual = 1
       self.lambda_adv = 0.05

    def training_step(self, input_images: torch.Tensor, target_images: torch.Tensor, input_ages: list, target_ages: list) -> dict:
       """
       Performs a single training step for the generator and discriminator.

       Args:
           input_images (torch.Tensor): Batch of input images.
           target_images (torch.Tensor): Batch of target images.
           input_ages (list): Input ages.
           target_ages (list): Target ages.

       Returns:
           dict: A dictionary containing the losses for the generator and discriminator.
       """
       # Note that the ages are divided by 100 before being passed to the model
       input_age_tensor = self.age_to_tensor(self.normalise_age(input_ages), 512)
       target_age_tensor = self.age_to_tensor(self.normalise_age(target_ages), 512)

       input_images = input_images.to(self.device)
       input_age_tensor = input_age_tensor.to(self.device)
       target_age_tensor = target_age_tensor.to(self.device)

       # Step 1: Train the discriminator
       self.optimizer_D.zero_grad()

       # Feed input images, input ages and target ages to the generator
       generator_input = torch.cat((input_images, input_age_tensor, target_age_tensor), dim=1)
       delta = self.generator(generator_input)
       # Add delta [-1,1] to original image
       fake_images = input_images + delta

       # Feed fake/real images to the discriminator
       fake_aux = torch.cat((fake_images, target_age_tensor), dim=1)
       fake_output = self.discriminator(fake_aux)
       real_output = self.discriminator(torch.cat((input_images, input_age_tensor), dim=1))

       # Feed real images with fake ages to the discriminator
       fake_ages = self.create_fake_ages(input_ages)
       fake_ages_tensor = self.age_to_tensor([self.normalise_age(x) for x in fake_ages], 512)
       fake_ages_output = self.discriminator(torch.cat((input_images.to(self.device), fake_ages_tensor.to(self.device)), dim=1))

       # Calculate discriminator loss and backpropagate
       errD_real, errD_fake, errD_fake_ages = self.discriminator_loss(real_output.squeeze(), fake_output.squeeze(), fake_ages_output.squeeze())
       D_loss = errD_real + errD_fake + errD_fake_ages
       D_loss.backward()
       self.optimizer_D.step()

       # Train the generator
       self.optimizer_G.zero_grad()

       delta = self.generator(generator_input)
       fake_images = input_images + delta

       fake_aux = torch.cat((fake_images, target_age_tensor), dim=1)
       fake_output = self.discriminator(fake_aux)

       # Calculate generator loss and backpropagate
       errG, l1_loss, perceptual_loss = self.generator_loss(fake_output.squeeze(), target_images, fake_images)
       G_loss = errG+l1_loss+perceptual_loss
       G_loss.backward()
       self.optimizer_G.step()

       # Update loss dictionary
       loss_dict={'G_adv': errG.item(), 'G_L1':l1_loss.item(), 'G_perc': perceptual_loss.item(),
                  'D_fake': errD_fake.item(), 'D_real':errD_real.item(), 'D_fake_ages': errD_fake_ages.item(),
                  'G_total':G_loss.item(), 'D_total':D_loss.item()}

       return loss_dict

    def discriminator_loss(self, real_output: torch.Tensor, fake_output: torch.Tensor, fake_ages_output: torch.Tensor) -> tuple:
       """
       Calculates the discriminator loss.

       Args:
           real_output (torch.Tensor): Output of the discriminator for real images.
           fake_output (torch.Tensor): Output of the discriminator for fake images.
           fake_ages_output (torch.Tensor): Output of the discriminator for real images with fake ages.

       Returns:
           tuple: A tuple containing the real loss, fake loss, and fake ages loss.
       """
       criterion = torch.nn.BCELoss()
       real_label = 1
       fake_label = 0

       errD_real = criterion(real_output.to(self.device),torch.full_like(real_output, real_label).to(self.device))/3
       errD_fake = criterion(fake_output.to(self.device), torch.full_like(fake_output, fake_label).to(self.device))/3
       errD_fake_ages = criterion(fake_ages_output.to(self.device), torch.full_like(fake_ages_output, fake_label).to(self.device))/3

       return errD_real, errD_fake, errD_fake_ages

    def generator_loss(self, fake_output: torch.Tensor, target_images: torch.Tensor, fake_images: torch.Tensor) -> tuple:
       """
       Calculates the generator loss.

       Args:
           fake_output (torch.Tensor): Output of the discriminator for fake images.
           target_images (torch.Tensor): Target images.
           fake_images (torch.Tensor): Fake images generated by the generator.

       Returns:
           tuple: A tuple containing the adversarial loss, L1 loss, and perceptual loss.
       """
       criterion = torch.nn.BCELoss()
       real_label = 1

       # GAN loss
       errG = criterion(fake_output.to(self.device), torch.full_like(fake_output, real_label).to(self.device))*self.lambda_adv

       # L1 loss
       l1_loss = torch.nn.L1Loss()(target_images.to(self.device), fake_images.to(self.device))*self.lambda_L1

       # Perceptual loss
       perceptual_loss = self.lpips(target_images.to(self.device),fake_images.to(self.device)).mean()*self.lambda_perceptual

       return errG, l1_loss, perceptual_loss

    def save_generator(self, save_path: str) -> None:
        """
        Saves the generator model to the specified path.

        Args:
            save_path (str): The path to save the generator model.
        """
        torch.save(self.generator.state_dict(), save_path)
        print(f"Generator model saved at {save_path}")

    def save_discriminator(self, save_path: str) -> None:
        """
        Saves the discriminator model to the specified path.

        Args:
            save_path (str): The path to save the discriminator model.
        """
        torch.save(self.discriminator.state_dict(), save_path)
        print(f"Discriminator model saved at {save_path}")

    def infer(self, input_images: torch.Tensor, input_ages: list, target_ages: list) -> torch.Tensor:
        """
        Generates the delta tensor from input images, input ages, and target ages.

        Args:
            input_images (torch.Tensor): Input images.
            input_ages (list): Input ages.
            target_ages (list): Target ages.

        Returns:
            torch.Tensor: The delta tensor.
        """
        self.generator.eval()
        input_age_tensor = self.age_to_tensor(self.normalise_age(input_ages), 512)
        target_age_tensor = self.age_to_tensor(self.normalise_age(target_ages), 512)
        generator_input = torch.cat((input_images, input_age_tensor, target_age_tensor), dim=1)
        with torch.no_grad():
            delta = self.generator(generator_input.to(self.device))
        return delta

    def create_fake_ages(self, ages: list) -> list:
        """
        Creates fake ages for the discriminator.

        Args:
            ages (list): A list of input ages.

        Returns:
            list: A list of fake ages.
        """
        random_ages = []
        possible_ages = self.age_list
        for i in range(len(ages)):
            age = random.choice(possible_ages)
            while age == ages[i]: # Ensure the fake age is different from the real age
                age = random.choice(possible_ages)
            random_ages.append(age)
        return random_ages

    @staticmethod
    def age_to_tensor(age: Union[list, float], size: int) -> torch.Tensor:
        """
        Converts an age or a list of ages to a tensor.

        Args:
            age (list | float): A single age or a list of ages.
            size (int): The size of the tensor.

        Returns:
            torch.Tensor: A tensor representation of the age(s).
        """
        if isinstance(age, float):
            tensor = torch.full((1, size, size), age).unsqueeze(0)
        else:
            repeated_lst = [torch.full((size, size), num) for num in age]
            # Stack the tensors along the first dimension to create a (len(age), 1, 512, 512) tensor
            tensor = torch.Tensor(torch.stack(repeated_lst, dim=0)).unsqueeze(1)
        return tensor

    @staticmethod
    def normalize(tensor: torch.Tensor, new_min: float = -1, new_max: float = 1) -> torch.Tensor:
        """
        Normalizes a tensor to the specified range.

        Args:
            tensor (torch.Tensor): The input tensor.
            new_min (float, optional): The minimum value of the new range. Defaults to -1.
            new_max (float, optional): The maximum value of the new range. Defaults to 1.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        current_min = torch.min(tensor)
        current_max = torch.max(tensor)
        # Apply linear transformation
        normalized_tensor = ((tensor - current_min) / (current_max - current_min)) * (new_max - new_min) + new_min
        return normalized_tensor

    @staticmethod
    def normalise_age(x: float) -> float:
        """
        Normalizes an age value by dividing it by 100.

        Args:
            x (float): The age value.

        Returns:
            float: The normalized age value.
        """
        return x / 100

