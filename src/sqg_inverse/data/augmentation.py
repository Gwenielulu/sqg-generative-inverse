import torch
import random


class RandomAxisFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            axis = random.choice([-2, -1])
            return torch.flip(x, dims=[axis])
        return x


class RandomAxisPermute:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            if x.ndim == 3:
                return x.permute(0, 2, 1)
            if x.ndim == 4:
                return x.permute(0, 1, 3, 2)
        return x


class RandomAxisRoll:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            if x.ndim == 3:
                H, W = x.shape[-2:]
                shift_h = random.randint(0, H - 1)
                shift_w = random.randint(0, W - 1)
                x = torch.roll(x, shifts=shift_h, dims=-2)
                x = torch.roll(x, shifts=shift_w, dims=-1)
            elif x.ndim == 4:
                H, W = x.shape[-2:]
                shift_h = random.randint(0, H - 1)
                shift_w = random.randint(0, W - 1)
                x = torch.roll(x, shifts=shift_h, dims=-2)
                x = torch.roll(x, shifts=shift_w, dims=-1)
        return x


class Compose:

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def get_augmentation(augment_types):
    if not augment_types:
        return None
    transforms = []
    for aug_type in augment_types:
        if aug_type == "random_axis_flip":
            transforms.append(RandomAxisFlip(p=0.5))
        elif aug_type == "random_axis_permute":
            transforms.append(RandomAxisPermute(p=0.5))
        elif aug_type == "random_axis_roll":
            transforms.append(RandomAxisRoll(p=0.5))
        elif aug_type in ("hflip", "vflip"):
            transforms.append(RandomAxisFlip(p=0.5))
        elif aug_type == "rotate":
            transforms.append(RandomAxisPermute(p=0.5))
        elif aug_type == "roll":
            transforms.append(RandomAxisRoll(p=0.5))
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")
    if transforms:
        return Compose(*transforms)
    else:
        return None
