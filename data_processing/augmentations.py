from PIL import Image, ImageEnhance
import torchvision.transforms as transforms


class CustomTransform(object):
    """Custom transformation class for data augmentation."""
    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float, rotation: float,
                 kernel_size: tuple[int, int], sigma: float, left: int, top: int) -> None:

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.rotation = rotation
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.left = left
        self.top = top

    def __call__(self, img: Image.Image) -> Image.Image:
        img = ImageEnhance.Brightness(img).enhance(self.brightness)
        img = ImageEnhance.Contrast(img).enhance(self.contrast)
        img = ImageEnhance.Color(img).enhance(self.saturation)
        img = transforms.functional.adjust_hue(img, self.hue)

        img = transforms.functional.rotate(img, self.rotation)
        img = self.gaussian_blur(img)

        img_width, img_height = img.size
        crop_size = int(4 / 5 * min(img_width, img_height))
        left, top = self.left, self.top
        img = img.crop((left, top, left + crop_size, top + crop_size))
        img = img.resize((img_width, img_height), Image.BICUBIC)

        return img


