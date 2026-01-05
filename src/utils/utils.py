import torch
import numpy as np

def denormalize(tensor, mean, std):
    """Denormalize tensor for visualization."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def denormalize_image(img_tensor, mean, std):
    """Denormalize image for visualization."""
    img = img_tensor.permute(1, 2, 0).numpy()
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    return img

def custom_collate_fn(batch):
    """Custom collate function to handle None masks in batches."""
    batch = list(zip(*batch))
    images = torch.stack(batch[0])
    masks = batch[1] # Keep as tuple/list to handle None
    labels = torch.tensor(batch[2])
    paths = batch[3]
    return images, masks, labels, paths