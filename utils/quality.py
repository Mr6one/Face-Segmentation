import numpy as np
import torch
from lib import *


def model_quality(model, dataset, device='cuda'):
    dice = []

    with torch.no_grad():
        for image, mask in dataset:
            image = image.to(device).unsqueeze(0)
            mask = mask.numpy().astype('int8')

            predicted_mask = model(image)
            predicted_mask = (predicted_mask >= 0) * 1
            predicted_mask = predicted_mask.squeeze(1).detach().cpu().numpy().astype('int8')

            dice.append(get_dice(mask, predicted_mask))

    return np.mean(dice)
