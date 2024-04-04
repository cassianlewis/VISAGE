import torch
import torch.optim as optim
from lpips import LPIPS
import random


class FRANTrainer:
    def __init__(self, generator, discriminator, age_list, device="cuda", batch_size=8):
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


    def training_step(self, epoch, input_images, target_images, input_ages, target_ages):

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

    def discriminator_loss(self, real_output, fake_output, fake_ages_output):
        criterion = torch.nn.BCELoss()
        real_label = 1
        fake_label = 0

        errD_real = criterion(real_output.to(self.device),torch.full_like(real_output, real_label).to(self.device))/3
        errD_fake = criterion(fake_output.to(self.device), torch.full_like(fake_output, fake_label).to(self.device))/3
        errD_fake_ages = criterion(fake_ages_output.to(self.device), torch.full_like(fake_ages_output, fake_label).to(self.device))/3

        return errD_real, errD_fake, errD_fake_ages

    def generator_loss(self, fake_output, target_images, fake_images):
        criterion = torch.nn.BCELoss()
        real_label = 1

        # GAN loss
        errG = criterion(fake_output.to(self.device), torch.full_like(fake_output, real_label).to(self.device))*self.lambda_adv
        # L1 loss
        l1_loss = torch.nn.L1Loss()(target_images.to(self.device), fake_images.to(self.device))*self.lambda_L1
        # Perceptual loss
        perceptual_loss = self.lpips(target_images.to(self.device),fake_images.to(self.device)).mean()*self.lambda_perceptual

        return errG, l1_loss, perceptual_loss

    def normalize(self, tensor, new_min=-1, new_max=1):
        # Function to normalize a tensor in the range of [0,1]
        current_min = torch.min(tensor)
        current_max = torch.max(tensor)

        # Apply linear transformation
        normalized_tensor = ((tensor - current_min) / (current_max - current_min)) * (new_max - new_min) + new_min

        return normalized_tensor

    def save_generator(self, save_path):
        torch.save(self.generator.state_dict(), save_path)
        print(f"Generator model saved at {save_path}")

    def save_discriminator(self, save_path):
        torch.save(self.discriminator.state_dict(), save_path)
        print(f"Discriminator model saved at {save_path}")

    def infer(self, input_images, input_ages, target_ages):
        # Function to generate delta from input images/age and target age
        self.generator.eval()
        input_age_tensor = self.age_to_tensor(self.normalise_age(input_ages), 512)
        target_age_tensor = self.age_to_tensor(self.normalise_age(target_ages), 512)
        generator_input = torch.cat((input_images, input_age_tensor, target_age_tensor), dim=1)
        with torch.no_grad():
            delta = self.generator(generator_input.to(self.device))
        return delta

    def create_fake_ages(self, ages):
        # Function to create fake ages for the discriminator
        random_ages=[]
        possible_ages = self.age_list
        for i in range(len(ages)):
            age = random.choice(possible_ages)
            while age == ages[i]:
                age = random.choice(possible_ages)
            random_ages.append(age)

        return random_ages

    def age_to_tensor(self, age, size):
        # Function to convert age float or list to a tensor
        if isinstance(age, float):
            tensor = torch.full((1, size, size), age).unsqueeze(0)

        else:
            repeated_lst = [torch.full((size, size), num) for num in age]

            # Stack the tensors along the first dimension to create a (len(age), 1, 512, 512) tensor
            tensor = torch.Tensor(torch.stack(repeated_lst, dim=0)).unsqueeze(1)
        return tensor

    def normalise_age(self, x):
        return x/100


