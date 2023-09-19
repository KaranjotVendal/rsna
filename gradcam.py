import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

import matplotlib.pyplot as plt
from matplotlib import colormaps

import numpy as np
import PIL

from config import config


class GradCAM():  
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        # defines two global scope variables to store our gradients and activations
        self.gradients = None
        self.activations = None
        
        self.back_hook = self.target_layer.register_full_backward_hook(self.backward_hook)
        self.frwd_hook = self.target_layer.register_forward_hook(self.forward_hook, prepend=False)
        
    def backward_hook(self, module, grad_input, grad_output):
        print('Backward hook running...')
        self.gradients = grad_output
        # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
        print(f'Gradients size: {self.gradients[0].size()}') 
        # We need the 0 index because the tensor containing the gradients comes
        # inside a one element tuple.

    def forward_hook(self, module, args, output):
        print('Forward hook running...')
        self.activations = output
        # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
        print(f'Activations size: {self.activations.size()}')

    def compute_cam(self, input_tensor, org, target_class=None):
        # Forward pass
        logits, probs = self.model(input_tensor.to(config.DEVICE), org)
        pred_idx = torch.argmax(logits, dim=1)

        print('starting backward pass')
        logits[:,pred_idx].backward()
        print('finished the backward pass')

        # pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients[0], dim=[0, 2, 3])

        # weight the channels by corresponding gradients
        for i in range(self.activations.size()[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()

        # relu on top of the heatmap
        heatmap = F.relu(heatmap)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        #plt.matshow(heatmap.detach())

        #remove hooks
        self.back_hook.remove()
        self.frwd_hook.remove()

        return heatmap
        
    def show_result(self, input_tensor, heatmap):
        # Create a figure and plot the first image
        fig, ax = plt.subplots()
        ax.axis('off') # removes the axis markers

        # First plot the original image
        ax.imshow(to_pil_image(input_tensor))

        # Resize the heatmap to the same size as the input image and defines
        # a resample algorithm for increasing image resolution
        # we need heatmap.detach() because it can't be converted to numpy array while
        # requiring gradients
        overlay = to_pil_image(heatmap.detach(), mode='F').resize((256,256), resample=PIL.Image.BICUBIC)

        # Apply any colormap you want
        cmap = colormaps['jet']
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

        # Plot the heatmap on the same axes, 
        # but with alpha < 1 (this defines the transparency of the heatmap)
        ax.imshow(overlay, alpha=0.2, interpolation='nearest')#, extent=extent)

        # Show the plot
        plt.show()
        plt.savefig('./plots/GradCAM/gradcam_img_.png', dpi=300)












































































































































#==============================================================================================================


class GradCAM:  
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.outputs = None

        # Hook the target layer to get gradients and activations
        self.hooks = []
        self.hooks.append(self.target_layer.register_forward_hook(self.save_output))
        self.hooks.append(self.target_layer.register_backward_hook(self.save_gradient))

    def save_output(self, module, input, output):
        self.outputs = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute_cam(self, input_tensor, org, target_category=None):
        # Forward pass
        logit, probs = self.model(input_tensor, org)
        
        if target_category is None:
            target_category = torch.argmax(probs).item()

        # Zero all other classes except target
        one_hot_output = torch.zeros_like(model_output)
        one_hot_output[0][target_category] = 1
        
        # Backward pass with specified target
        self.model.zero_grad()
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        
        # Average gradients spatially
        weights = torch.mean(self.gradients, dim=(2, 3))[0, :]
        
        # Create empty CAM
        cam = torch.zeros_like(self.outputs[0], memory_format=torch.contiguous_format)
        
        # Compute CAM by taking a weighted average of feature maps using the weights computed above
        for i, w in enumerate(weights):
            cam += w * self.outputs[0, i, :, :]

        # Resize the CAM to the size of the input image and clip negative values
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2:], mode='bilinear', align_corners=True).squeeze().cpu().detach().numpy()
        
        # Normalize the CAM
        cam -= cam.min()
        cam /= cam.max()

        # Clean up
        for hook in self.hooks:
            hook.remove()
        
        return cam

# This GradCAM class can be used to generate CAM for a given input and model. You'll need to specify the target layer of the model when initializing GradCAM.



###################################################################################################################################################################


def make_gradcam_heatmap(img_tensor, model, last_conv_layer_name, pred_index=None):
    """
    Generate class activation heatmap using PyTorch.
    """
    # Ensure the image tensor's gradients can be computed
    img_tensor.requires_grad_(True)
    
    # Fetch the last convolutional layer from the model by its name
    last_conv_layer = dict(model.named_modules())[last_conv_layer_name]
    
    # Forward hook to fetch the output of the last convolutional layer
    activations = []
    def hook_fn(module, input, output):
        activations.append(output)
    hook = last_conv_layer.register_forward_hook(hook_fn)
    
    # Forward pass
    preds = model(img_tensor)
    
    # If pred_index is not provided, take the index of the highest prediction
    if pred_index is None:
        pred_index = torch.argmax(preds[0]).item()
    
    # Only keep the prediction we're interested in for the gradient computation
    target_class = preds[0][pred_index]
    
    # Backward pass
    model.zero_grad()
    target_class.backward()
    
    # Remove the hook after use
    hook.remove()
    
    # Get the gradients and activations
    grads = activations[0].grad[0]
    activations = activations[0][0]

    # Global average pooling of the gradients
    pooled_grads = torch.mean(grads, dim=(1, 2))
    
    # Multiply each channel in the feature map array by 'how important this channel is'
    # This gives the heatmap class activation
    heatmap = torch.matmul(activations.permute(1, 2, 0), pooled_grads.unsqueeze(-1))
    heatmap = heatmap.permute(2, 0, 1).squeeze(0)
    
    # Normalize between 0 and 1 and apply ReLU
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    
    return heatmap.detach().numpy()


def get_resized_heatmap(heatmap, shape):
    """Resize heatmap to shape"""
    # Rescale heatmap to a range 0-255
    upscaled_heatmap = np.uint8(255 * heatmap)

    upscaled_heatmap = zoom(
        upscaled_heatmap,
        (
            shape[0] / upscaled_heatmap.shape[0],
            shape[1] / upscaled_heatmap.shape[1],
            shape[2] / upscaled_heatmap.shape[2],
        ),
    )

    return upscaled_heatmap


resized_heatmap = get_resized_heatmap(heatmap, input_volume.shape)



fig, ax = plt.subplots(1, 2, figsize=(10, 20))

# Convert PyTorch tensors to numpy arrays for visualization
input_volume_np = input_volume.squeeze().numpy()

ax[0].imshow(input_volume_np[:, :, 30], cmap='bone')
img0 = ax[1].imshow(input_volume_np[:, :, 30], cmap='bone')
img1 = ax[1].imshow(resized_heatmap[:, :, 30], cmap='jet', alpha=0.3, extent=img0.get_extent())
plt.show()