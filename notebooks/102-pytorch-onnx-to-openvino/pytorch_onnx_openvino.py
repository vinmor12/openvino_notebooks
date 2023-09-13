# FIFTH EXAMPLE
# This example demonstrates step-by-step instructions on how to do inference
# on a PyTorch semantic segmentation model, using OpenVINO Runtime.
# First, the PyTorch model is exported in ONNX format and then converted
# to OpenVINO IR. Then the respective ONNX and OpenVINO IR models are loaded
# into OpenVINO Runtime to show model predictions.
# We will use LR-ASPP model with MobileNetV3 backbone.

# IMPORTS --------------------------------------------------------------------------------------------------------------------
print("Start Imports ")
import sys
import time
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import torch
from IPython.display import Markdown, display
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
from openvino.runtime import Core
sys.path.append("../utils")
from notebook_utils import segmentation_map_to_image, viz_result_image, SegmentationMap, Label, download_file
print("Modules Have Been Imported")

# SETTINGS ---------------------------------------------------------------------------------------------------------------------
# Taking input from console
print("Write Directory Name of Model:\n(example: model)")
DIRECTORY_NAME = input()
print("Write Model Name:\n(example: /lraspp_mobilenet_v3_large)")
MODEL_NAME = input()
# Default Values
IMAGE_WIDTH = 780
IMAGE_HEIGHT =  520
# Paths where file “.pt” will be stored.
BASE_MODEL_NAME = DIRECTORY_NAME + MODEL_NAME
weights_path = Path(BASE_MODEL_NAME + ".pt")
# Paths where ONNX and OpenVINO IR models will be stored.
onnx_path = weights_path.with_suffix('.onnx')
if not onnx_path.parent.exists():
    onnx_path.parent.mkdir()
ir_path = onnx_path.with_suffix(".xml")
print("Path created!") 
    
# LOAD MODEL ---------------------------------------------------------------------------------------------------------------------
# Typical steps for getting a pre-trained model:
# 1. Create instance of model class
# 2. Load checkpoint state dict, which contains pre-trained model weights
# 3. Turn model to evaluation for switching some operations to inference mode    
print("Downloading the LRASPP MobileNetV3 model (if it has not been downloaded already)...")
download_file(LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.url, filename=weights_path.name, directory=weights_path.parent)
# create model object
model = lraspp_mobilenet_v3_large()
# read state dict, use map_location argument to avoid a situation where weights are saved in cuda (which may not be unavailable on the system)
state_dict = torch.load(weights_path, map_location='cpu')
# load state dict to model
model.load_state_dict(state_dict)
# switch model from training to inference mode
model.eval()
print("Loaded PyTorch LRASPP MobileNetV3 model!")

# CONVERT PYTORCH MODEL TO ONNX ------------------------------------------------------------------------------------------------
# OpenVINO supports PyTorch models that are exported in ONNX format
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    if not onnx_path.exists():
        dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
        )
        print(f"ONNX model exported to {onnx_path}.")
    else:
        print(f"ONNX model {onnx_path} already exists.")

# CONVERT ONNX MODEL TO OPENVINO IR --------------------------------------------------------------------------------------------
# Use Model Optimizer to convert the ONNX model to OpenVINO IR with FP16 precision.
# The models are saved inside the current directory.
# Construct the command for Model Optimizer.
mo_command = f"""mo
                 --input_model "{onnx_path}"
                 --compress_to_fp16
                 --output_dir "{ir_path.parent}"
                 """
mo_command = " ".join(mo_command.split())
print("Model Optimizer command to convert the ONNX model to OpenVINO:")
display(Markdown(f"`{mo_command}`"))
if not ir_path.exists():
    print("Exporting ONNX model to IR... This may take a few minutes.")
    mo_result = os.system(mo_command)
else:
    print(f"IR model {ir_path} already exists.")

# LOAD AND PREPROCESS AN INPUT IMAGE ----------------------------------------------------------------------------------------
#Images need to be normalized before propagating through the network
# Taking input from console
print("Write Image Path:\n(example: ../data/image/coco.jpg)")
image_filename = input()
# Def normalize function
def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image to the given mean and standard deviation
    for ImageNet data.
    """
    image = image.astype(np.float32)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image /= 255.0
    image -= mean
    image /= std
    return image
# Preprocess image
image = cv2.cvtColor(cv2.imread(image_filename), cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
normalized_image = normalize(resized_image)
# Convert the resized images to network input shape.
input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
normalized_input_image = np.expand_dims(np.transpose(normalized_image, (2, 0, 1)), 0)
# Plot image
print("Input Image")
plt.imshow(image)
print("Close the Image to Continue ...")
plt.show()
# Plot normalized/resized image
# Clipping input data to the valid range for imshow with RGB data
# [0..1] for floats or [0..255] for integers
print("Normalized Image")
plt.imshow(cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB))
print("Close the Image to Continue ...")
plt.show()

# COLOR CODING ---------------------------------------------------------------------------------------------------------------
# Color coding can be applied to each label for more convenient visualization.
voc_labels = [
    Label(index=0, color=(0, 0, 0), name="background"),
    Label(index=1, color=(128, 0, 0), name="aeroplane"),
    Label(index=2, color=(0, 128, 0), name="bicycle"),
    Label(index=3, color=(128, 128, 0), name="bird"),
    Label(index=4, color=(0, 0, 128), name="boat"),
    Label(index=5, color=(128, 0, 128), name="bottle"),
    Label(index=6, color=(0, 128, 128), name="bus"),
    Label(index=7, color=(128, 128, 128), name="car"),
    Label(index=8, color=(64, 0, 0), name="cat"),
    Label(index=9, color=(192, 0, 0), name="chair"),
    Label(index=10, color=(64, 128, 0), name="cow"),
    Label(index=11, color=(192, 128, 0), name="dining table"),
    Label(index=12, color=(64, 0, 128), name="dog"),
    Label(index=13, color=(192, 0, 128), name="horse"),
    Label(index=14, color=(64, 128, 128), name="motorbike"),
    Label(index=15, color=(192, 128, 128), name="person"),
    Label(index=16, color=(0, 64, 0), name="potted plant"),
    Label(index=17, color=(128, 64, 0), name="sheep"),
    Label(index=18, color=(0, 192, 0), name="sofa"),
    Label(index=19, color=(128, 192, 0), name="train"),
    Label(index=20, color=(0, 64, 128), name="tv monitor")
]
VOCLabels = SegmentationMap(voc_labels)
    
# ONNX INFERENCE ------------------------------------------------------------------------------------------------------------
# ONNX Model in OpenVINO Runtime
# Load the network to OpenVINO Runtime.
ie = Core()
model_onnx = ie.read_model(model=onnx_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")
output_layer_onnx = compiled_model_onnx.output(0)
# Run inference on the input image.
res_onnx = compiled_model_onnx([normalized_input_image])[output_layer_onnx]
# Convert the network result to a segmentation map and display the result.
result_mask_onnx = np.squeeze(np.argmax(res_onnx, axis=1)).astype(np.uint8)
# Plot Mask
print("ONNX Model Result:")
plt.imshow(segmentation_map_to_image(result_mask_onnx, VOCLabels.get_colormap()))
print("Close the Image to Continue ...")
plt.show()

# IR INFERENCE ------------------------------------------
# OpenVINO IR Model in OpenVINO Runtime
# Load the network in OpenVINO Runtime.
ie = Core()
model_ir = ie.read_model(model=ir_path)
compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")
# Get input and output layers.
output_layer_ir = compiled_model_ir.output(0)
# Run inference on the input image.
res_ir = compiled_model_ir([normalized_input_image])[output_layer_ir]
result_mask_ir = np.squeeze(np.argmax(res_ir, axis=1)).astype(np.uint8)
# Plot Mask
print("OpenVINO IR Model Result:")
plt.imshow(segmentation_map_to_image(result_mask_ir, VOCLabels.get_colormap()))
print("Close the Image to Continue ...")
plt.show()

# PYTORCH INFERENCE -----------------------------------------------
# Do inference on the PyTorch model to verify that the output visually looks the same
# as the output on the ONNX/OpenVINO IR models
model.eval()
with torch.no_grad():
    result_torch = model(torch.as_tensor(normalized_input_image).float())
result_mask_torch = torch.argmax(result_torch['out'], dim=1).squeeze(0).numpy().astype(np.uint8)
# Plot Mask
print("PyTorch Model Result:")
plt.imshow(segmentation_map_to_image(result_mask_torch, VOCLabels.get_colormap()))
print("Close the Image to Continue ...")
plt.show()

# PERFORMANCE COMPARISON ------------------------------------------
# Measure the time it takes to do inference on twenty images. This gives an indication
# of performance.
# Taking input from console
print("Write Number of Images for comparison:\n(example: 100)")
num_images = int(input())
print("Start Comparison ... ")
# Performance PyTorch (CPU)
with torch.no_grad():
    start = time.perf_counter()
    for _ in range(num_images):
        model(torch.as_tensor(input_image).float())
    end = time.perf_counter()
    time_torch = end - start
print(
    f"PyTorch model on CPU: {time_torch/num_images:.3f} seconds per image, "
    f"FPS: {num_images/time_torch:.2f}"
)
# Performance ONNX (CPU)
start = time.perf_counter()
for _ in range(num_images):
    compiled_model_onnx([normalized_input_image])
end = time.perf_counter()
time_onnx = end - start
print(
    f"ONNX model in OpenVINO Runtime/CPU: {time_onnx/num_images:.3f} "
    f"seconds per image, FPS: {num_images/time_onnx:.2f}"
)
# Performance OpenVINO IR (CPU)
start = time.perf_counter()
for _ in range(num_images):
    compiled_model_ir([input_image])
end = time.perf_counter()
time_ir = end - start
print(
    f"OpenVINO IR model in OpenVINO Runtime/CPU: {time_ir/num_images:.3f} "
    f"seconds per image, FPS: {num_images/time_ir:.2f}"
)
# Performance ONNX on GPU (if it is available)
if "GPU" in ie.available_devices:
    compiled_model_onnx_gpu = ie.compile_model(model=model_onnx, device_name="GPU")
    start = time.perf_counter()
    for _ in range(num_images):
        compiled_model_onnx_gpu([input_image])
    end = time.perf_counter()
    time_onnx_gpu = end - start
    print(
        f"ONNX model in OpenVINO/GPU: {time_onnx_gpu/num_images:.3f} "
        f"seconds per image, FPS: {num_images/time_onnx_gpu:.2f}"
    )
# Performance OpenVINO IR on GPU (if it is available)
    compiled_model_ir_gpu = ie.compile_model(model=model_ir, device_name="GPU")
    start = time.perf_counter()
    for _ in range(num_images):
        compiled_model_ir_gpu([input_image])
    end = time.perf_counter()
    time_ir_gpu = end - start
    print(
        f"IR model in OpenVINO/GPU: {time_ir_gpu/num_images:.3f} "
        f"seconds per image, FPS: {num_images/time_ir_gpu:.2f}"
    )

# Vincenzo

    


