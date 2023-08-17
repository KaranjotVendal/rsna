import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image

from models import RACNet
from config import config
from utils import load_image

def gradcam(images, test_loader, fold):
    #images
    model = RACNet(config.MODEL, config.NUM_CLASSES)
    model.to(device)
    checkpoint = torch.load(f'checkpoints/best-model-{fold}.pth')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # The target for the CAM is the Bear category.
    # As usual for classication, the target is the logit output
    # before softmax, for that category.
    targets = [ClassifierOutputTarget(1)]
    target_layers = [model.cnn.layer4[-1]]
    
    for idx, batch in enumerate(test_loader):
   
        with torch.no_grad():
            features = batch['X'].to(config.DEVICE)
            #targets = batch['y'].to(self.device)
            org = batch['org']

            #logits, probs = self.model(features, org)
            #predicted_class = probs.argmax(dim=1)
            #test_targets.append(targets)
            #preds.append(probs.detach())

            with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:
                grayscale_cams = cam(input_tensor=features, targets=targets)
                cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
            cam = np.uint8(255*grayscale_cams[0, :])
            cam = cv2.merge([cam, cam, cam])
            images = np.hstack((np.uint8(255*img), cam , cam_image))
            Image.fromarray(images)




























# defines two global scope variables to store our gradients and activations
gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
  global gradients # refers to the variable in the global scope
  print('Backward hook running...')
  gradients = grad_output
  # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
  print(f'Gradients size: {gradients[0].size()}') 
  # We need the 0 index because the tensor containing the gradients comes
  # inside a one element tuple.

def forward_hook(module, args, output):
  global activations # refers to the variable in the global scope
  print('Forward hook running...')
  activations = output
  # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
  print(f'Activations size: {activations.size()}')


model = RACNet(config.MODEL, config.NUM_CLASSES)
checkpoint = torch.load(os.path.join(settings['MODEL_CHECKPOINT_DIR'], f'best-model-{fold}.pth'))
model.load_state_dict(checkpoint["model_state_dict"])
backward_hook = model.cnn.layer4[-1].register_full_backward_hook(backward_hook, prepend=False)
forward_hook = model.cnn.layer4[-1]3.register_forward_hook(forward_hook, prepend=False)

img = load_image('data/reduced_dataset/00002/FLAIR/Image-433.png')
'./data/reduced_dataset/',
                        xtrain,  
                        ytrain,
                        n_slices=254,
                        img_size=112,
                        transform=None
                            




def gradcam():
    






















import warnings
warnings.filterwarnings('ignore')
from torchvision import models
import numpy as np
import cv2
import requests
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image

model = models.resnet50(pretrained=True)
model.eval()
image_url = "https://th.bing.com/th/id/R.94b33a074b9ceeb27b1c7fba0f66db74?rik=wN27mvigyFlXGg&riu=http%3a%2f%2fimages5.fanpop.com%2fimage%2fphotos%2f31400000%2fBear-Wallpaper-bears-31446777-1600-1200.jpg&ehk=oD0JPpRVTZZ6yizZtGQtnsBGK2pAap2xv3sU3A4bIMc%3d&risl=&pid=ImgRaw&r=0"
img = np.array(Image.open(requests.get(image_url, stream=True).raw))
img = cv2.resize(img, (224, 224))
img = np.float32(img) / 255
input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# The target for the CAM is the Bear category.
# As usual for classication, the target is the logit output
# before softmax, for that category.
targets = [ClassifierOutputTarget(295)]
target_layers = [model.layer4]
with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])
images = np.hstack((np.uint8(255*img), cam , cam_image))
Image.fromarray(images)










#=================================================================================
'''gradcam implementation by chatgpt'''

import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None
        
        # Hook the feature maps and gradients at the target layer
        self.hooks = []
        self.hooks.append(self.target_layer.register_forward_hook(self.save_feature_maps))
        self.hooks.append(self.target_layer.register_backward_hook(self.save_gradients))
    
    def save_feature_maps(self, module, input, output):
        """Function to hook the feature maps of the target layer"""
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_in, grad_out):
        """Function to hook the gradients of the target layer"""
        self.gradients = grad_out[0].detach()
        
    def compute_heatmap(self, input_tensor, class_idx=None):
        """
        Compute the Grad-CAM heatmap for a given input tensor and target class index.
        
        Args:
        - input_tensor (torch.Tensor): The input tensor for the model. Shape (1, C, H, W)
        - class_idx (int): The target class index. If None, use the predicted class.
        
        Returns:
        - heatmap (numpy.ndarray): The computed Grad-CAM heatmap.
        """
        # Zero out the previous gradients and feature maps
        self.feature_maps = None
        self.gradients = None
        
        # Forward pass
        logits = self.model(input_tensor)
        
        # If no class index provided, use the predicted class
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
            
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for the specific class index
        logits[0, class_idx].backward(retain_graph=True)
        
        # Compute the weights as the mean of the gradients along the spatial dimensions
        weights = torch.mean(self.gradients,









#======================================================================================================================


import torch.nn.functional as F

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

    def compute_cam(self, input_tensor, target_category=None):
        # Forward pass
        model_output = self.model(input_tensor)
        
        if target_category is None:
            target_category = torch.argmax(model_output).item()

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












    
