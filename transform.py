"""

Author: Honggu Liu
"""
from torchvision import transforms

transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),

        transforms.Normalize([0.5]*1, [0.5]*1)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 1, [0.5] * 1)
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 1, [0.5] * 1)
    ]),
}
