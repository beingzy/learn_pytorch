import torch
import numpy as np
import torchvision.transforms as T

from PIL import Image


def get_cart_location(env, screen_width):
    """
    """
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


def get_screen(env, resize, device):
    """ retruned screen requested by gym is 400x600x3, but this is 
        sometimes larger such as 800x1200x3. Transpose it into torch
        order (CHW)
    """
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)

    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(
            cart_location - view_width // 2, 
            cart_location + view_width // 2)

    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)
