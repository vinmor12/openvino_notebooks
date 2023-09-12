# FIRST EXAMPLE
# This example shows how to do inference with an image classification model.
# A pre-trained MobileNetV3 model from Open Model Zoo is used in this example.

# IMPORTS ---------------------------------------------------------------------
print("Start Imports ")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
print("Modules Have Been Imported ")

# LOAD MODEL ------------------------------------------------------------------
print("\nWrite the Path of Model:\n(example: model/v3-small_224_1.0_float.xml)")
model_path = input()
# Initialize OpenVINO Runtime
ie = Core()
# Read Model
model = ie.read_model(model=model_path)
# Compile Model
print("\nWrite the target Device:\n(example: CPU)")
target_device = input()
compiled_model = ie.compile_model(model=model, device_name=target_device)
output_layer = compiled_model.output(0)

#LOAD IMAGE -------------------------------------------------------------------
print("\nWrite the Path of Input Image:\n(example: ../data/image/coco.jpg)")
image_path = input()
# The MobileNet model expects images in RGB format.
image = cv2.cvtColor(cv2.imread(filename=image_path), code=cv2.COLOR_BGR2RGB)
# Resize to MobileNet image shape.
input_image = cv2.resize(src=image, dsize=(224, 224))
# Reshape to model input shape.
input_image = np.expand_dims(input_image, 0)
plt.imshow(image);
print("\nClose the Image to Continue ...")
plt.show()

# INFERENCE -------------------------------------------------------------------
result_infer = compiled_model([input_image])[output_layer]
result_index = np.argmax(result_infer)
# Convert the inference result to a class name.
print("\nWrite the Path of Imagenet Classes:\n(example: ../data/datasets/imagenet/imagenet_2012.txt)")
imagenet_classes_path = input()
imagenet_classes = open(imagenet_classes_path).read().splitlines()
imagenet_classes = ['background'] + imagenet_classes
imagenet_classes[result_index]
print("\nResult: ")
print(imagenet_classes[result_index])

# Vincenzo
