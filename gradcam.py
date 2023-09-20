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
        
        print(f'prediction: {pred_idx}')

        logits[:,pred_idx].backward()
        
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

        heatmap = heatmap.cpu().detach().numpy()
        input_tensor = input_tensor.squeeze(0)
        input_tensor = input_tensor.cpu().detach().numpy()

        return heatmap, input_tensor
        
    #t be improved and tested    
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
        #plt.savefig('./plots/GradCAM/gradcam_{}img_{img}.png', dpi=300)