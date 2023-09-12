# SECOND EXAMPLE
# This example shows how to do inference with a segmentation model.
# In this tutorial, a pre-trained road-segmentation-adas-0001 model
# from the Open Model Zoo is used

# IMPORTS ---------------------------------------------------------------------
print("Start Imports ")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from openvino.runtime import Core
sys.path.append("../utils")
from notebook_utils import segmentation_map_to_image
print("Modules Have Been Imported ")

# LOAD MODEL ------------------------------------------------------------------
print("\nWrite the Path of Model:\n(example: model/road-segmentation-adas-0001.xml)")
model_path = input()
# Initialize OpenVINO Runtime
ie = Core()
# Read Model
model = ie.read_model(model=model_path)
# Compile Model
print("\nWrite the target Device:\n(example: CPU)")
target_device = input()
compiled_model = ie.compile_model(model=model, device_name=target_device)
input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output(0)

# LOAD INPUT IMAGE ------------------------------------------------------------
print("\nWrite the Path of Input Image:\n(example: ../data/image/empty_road_mapillary.jpg)")
image_path = input()
# The segmentation network expects images in BGR format
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_h, image_w, _ = image.shape
# Define batch size, number of channels, height & width
N, C, H, W = input_layer_ir.shape
# Resize image
resized_image = cv2.resize(image, (W, H))
# Reshape to the network input shape
input_image = np.expand_dims(
    resized_image.transpose(2, 0, 1), 0
)
# Plot RGB image
plt.imshow(rgb_image)
print("\nClose the Image to Continue ...")
plt.show()

# INFERENCE -------------------------------------------------------------------
# Run the inference
result = compiled_model([input_image])[output_layer_ir]
# Prepare data for visualization
segmentation_mask = np.argmax(result, axis=1)
plt.imshow(segmentation_mask.transpose(1, 2, 0))
print("\nClose the Image to Continue ...")
plt.show()

# VISUALIZE DATA --------------------------------------------------------------
# Define colormap, each color represents a class
colormap = np.array([[68, 1, 84], [48, 103, 141], [53, 183, 120], [199, 216, 52]])
# Define the transparency of the segmentation mask on the photo
alpha = 0.3
# Use function from notebook_utils.py to transform mask to an RGB image
mask = segmentation_map_to_image(segmentation_mask, colormap)
resized_mask = cv2.resize(mask, (image_w, image_h))
# Create an image with mask
image_with_mask = cv2.addWeighted(resized_mask, alpha, rgb_image, 1 - alpha, 0)
# Display an image
plt.imshow(image_with_mask)
print("\nFinal Result ...")
plt.show()

# Vincenzo
