import tonic
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from skimage.transform import rescale, resize, downscale_local_mean
import torch
import torch.nn as nn
import sinabs.layers as sl
import torchvision
from scipy.special import iv
matplotlib.use('TkAgg')


def zero_2pi_tan(x, y):
    """
    Compute the angle in radians between the positive x-axis and the point (x, y),
    ensuring the angle is in the range [0, 2π].

    Args:
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.

    Returns:
        angle (float): Angle in radians, between 0 and 2π.
    """
    angle = np.arctan2(y, x) % (2 * np.pi)  # Get the angle in radians and wrap it in the range [0, 2π]
    return angle

def vm_filter(theta, scale, rho=0.1, r0=0, thick=0.5, offset=(0, 0)):
    """Generate a Von Mises filter with r0 shifting and an offset."""
    height, width = scale, scale
    vm = np.empty((height, width))
    offset_x, offset_y = offset

    for x in range(width):
        for y in range(height):
            # Shift X and Y based on r0 and offset
            X = (x - width / 2) + r0 * np.cos(theta) - offset_x * np.cos(theta)
            Y = (height / 2 - y) + r0 * np.sin(theta) - offset_y * np.sin(theta)  # Inverted Y for correct orientation
            r = np.sqrt(X**2 + Y**2)
            angle = zero_2pi_tan(X, Y)

            # Compute the Von Mises filter value
            vm[y, x] = np.exp(thick*rho * r0 * np.cos(angle - theta)) / iv(0, r - r0)
    # normalise value between -1 and 1
    # vm = vm / np.max(vm)
    # vm = vm * 2 - 1
    return vm


def create_vm_filters(thetas, size, rho, r0, thick, offset):
    """
    Create a set of Von Mises filters with different orientations.

    Args:
        thetas (np.ndarray): Array of angles in radians.
        size (int): Size of the filter.
        rho (float): Scale coefficient to control arc length.
        r0 (int): Radius shift from the center.

    Returns:
        filters (list): List of Von Mises filters.
    """
    filters = []
    for theta in thetas:
        filter = vm_filter(theta, size, rho=rho, r0=r0, thick=thick, offset=offset)
        filter = rescale(filter, fltr_resize_perc, anti_aliasing=False)
        filters.append(filter)
    filters = torch.tensor(np.stack(filters).astype(np.float32))
    return filters


def network_init(filters):
    """
    Initialize a neural network with a single convolutional layer using von Mises filters.

    Args:
        filters (torch.Tensor): Filters to be loaded into the convolutional layer.

    Returns:
        net (nn.Sequential): A simple neural network with one convolutional layer.
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # Define a sequential network with a Conv2D layer followed by an IAF layer
    net = nn.Sequential(
        nn.Conv2d(1, 8,  (filters.shape[1], filters.shape[2]), stride=1, bias=False),
        sl.LIF(tau_mem=tau_mem),
    )
    # Load the filters into the network weights
    net[0].weight.data = filters.unsqueeze(1).to(device)
    net[1].v_mem = net[1].tau_mem * net[1].v_mem.to(device)
    return net


def run_attention(window, net, device):
    # Create resized versions of the frames
    resized_frames = [torchvision.transforms.Resize((int(window.shape[2] / pyr), int(window.shape[1] / pyr)))(
        torch.from_numpy(window)) for pyr in range(1, num_pyr + 1)]
    # Process frames in batches
    batch_frames = torch.stack(
        [torchvision.transforms.Resize((window.shape[2] , window.shape[1]))(frame) for frame in resized_frames]).type(torch.float32)
    batch_frames = batch_frames.to(device)  # Move to GPU if available
    output_rot = net(batch_frames)
    # Sum the outputs over rotations and scales
    salmap = torch.sum(torch.sum(output_rot, dim=1, keepdim=True), dim=0, keepdim=True).squeeze().type(torch.float32)
    salmax_coords = np.unravel_index(torch.argmax(salmap).cpu().numpy(), salmap.shape)
    # normalise salmap for visualization
    salmap = salmap.detach().cpu()
    salmap = np.array((salmap - salmap.min()) / (salmap.max() - salmap.min()) * 255)
    # rescale salmap to the original size
    # salmap = resize(salmap, (window.shape[1], window.shape[2]), anti_aliasing=False)
    return salmap,salmax_coords

# Path to your dataset
path = '/Users/giuliadangelo/workspace/code/ISACT2024/'
labelled_data = '/Users/giuliadangelo/workspace/code/foveated-vision/tutorials/DVSGestureLabelled/'

dvs_training = tonic.datasets.DVSGesture(path, train=True)


# Visual attention paramters
size = 10  # Size of the kernel
r0 = 4  # Radius shift from the center
rho = 0.1  # Scale coefficient to control arc length
theta = np.pi * 3 / 2  # Angle to control the orientation of the arc
thick = 3  # thickness of the arc
offsetpxs = 0 #size / 2
offset = (offsetpxs, offsetpxs)
fltr_resize_perc = [2, 2]
num_pyr = 3
# Create Von Mises (VM) filters with specified parameters
# The angles are generated in radians, ranging from 0 to 2π in steps of π/4
thetas = np.arange(0, 2 * np.pi, np.pi / 4)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
tau_mem = 0.01


# loading attention kernels
filters_attention = create_vm_filters(thetas, size, rho, r0, thick, offset)
# plot_filters(filters_attention, thetas)


# Initialize the attention network with the loaded filters
netattention = network_init(filters_attention)


time_window=1000 #1 ms

#loop over all the users in dvs_training
for i in range(0, len(dvs_training)):
    events, npys = dvs_training[i]  # First user data
    # create forlder for each user
    folder = dvs_training.data[i].split('user')[1].split('/')[0]
    print('Folder: '+ folder+' and i: '+str(i))
    if not os.path.exists(labelled_data+folder):
        os.makedirs(labelled_data+folder)
    name_txtfile = f"{labelled_data+folder}/roi_data_{dvs_training.data[i].split('user')[1].split('/')[1].replace('.', '')}.txt"
    # Frame transform settings
    transform = tonic.transforms.ToFrame(
        sensor_size=dvs_training.sensor_size,
        time_window=time_window  # 30 ms
    )
    frames = transform(events)
    for frame in frames:
        salmap, salmap_coords = run_attention(frame, netattention, device)
        plt.imshow(salmap)
        plt.show()
        break


print('end')