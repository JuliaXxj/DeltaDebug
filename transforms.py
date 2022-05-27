from torchvision import transforms

from custom_transform import *


V = 0.5

TRANSFORMS = {
    "Original": transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "EraseUpperHalf": transforms.Compose([
        transforms.ToTensor(),
        EraseHalf(part='upper', value=V)
        # transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "EraseLowerHalf": transforms.Compose([
        transforms.ToTensor(),
        EraseHalf(part='lower', value=V)
        # transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "EraseLeftHalf": transforms.Compose([
        transforms.ToTensor(),
        EraseHalf(part='left', value=V)
        # transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "EraseRightHalf": transforms.Compose([
        transforms.ToTensor(),
        EraseHalf(part='right', value=V)
        # transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "KeepCenter75": transforms.Compose([
        transforms.ToTensor(),
        KeepCenter(percent=0.75, value=V)
        # transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "KeepCenter50": transforms.Compose([
        transforms.ToTensor(),
        KeepCenter(percent=0.5, value=V)
        # transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "KeepCenter25": transforms.Compose([
        transforms.ToTensor(),
        KeepCenter(percent=0.25, value=V)
        # transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "MaskEveryOtherPixel":transforms.Compose([
        transforms.ToTensor(),
        MaskEveryOtherPixel(value=V)
        # transforms.Normalize((0.1307,), (0.3081,))
    ]),
}

