# SIXTH EXAMPLE
# This example shows how to optimize an image classification model.
# A pre-trained MobileNetV3 model from Open Model Zoo is used in this example.

# IMPORTS ---------------------------------------------------------------------
print("Start Imports ")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from openvino.runtime import Core
print("Modules Have Been Imported ")

# LOAD MODEL ------------------------------------------------------------------
print("\nWrite the Path of Model FP32:\n(example: public/mobilenet-v3-small-1.0-224-tf/FP32/mobilenet-v3-small-1.0-224-tf.xml)")
model_path_32 = input()
print("\nWrite the Path of Model FP16:\n(example: public/mobilenet-v3-small-1.0-224-tf/FP16/mobilenet-v3-small-1.0-224-tf.xml)")
model_path_16 = input()
print("\nWrite the Path of Quantized Model:\n(example: results\model_name_DefaultQuantization/2023-09-14_11-48-49/optimized/mobilenet_v3_small_q.xml)")
model_path_q = input()
# Initialize OpenVINO Runtime
ie = Core()
# Read Model
model_32 = ie.read_model(model=model_path_32)
model_16 = ie.read_model(model=model_path_16)
model_q = ie.read_model(model=model_path_q)
# Compile Model
print("\nWrite the target Device:\n(example: CPU)")
target_device = input()
compiled_model_32 = ie.compile_model(model=model_32, device_name=target_device)
compiled_model_16 = ie.compile_model(model=model_16, device_name=target_device)
compiled_model_q = ie.compile_model(model=model_q, device_name=target_device)
output_layer_32 = compiled_model_32.output(0)
output_layer_16 = compiled_model_16.output(0)
output_layer_q = compiled_model_q.output(0)

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
print("... Close the Image to Continue ...")
plt.show()

# INFERENCE -------------------------------------------------------------------
start = time.perf_counter()
result_infer_32 = compiled_model_32([input_image])[output_layer_32]
end = time.perf_counter()
time_ir = end - start
result_index_32 = np.argmax(result_infer_32)
print("\nResult Index Model FP32:", result_index_32)
print("Infer Time Model FP32:", time_ir)
start = time.perf_counter()
result_infer_16 = compiled_model_16([input_image])[output_layer_16]
end = time.perf_counter()
time_ir = end - start
result_index_16 = np.argmax(result_infer_16)
print("\nResult Index Model FP16:", result_index_16)
print("Infer Time Model FP16:", time_ir)
start = time.perf_counter()
result_infer_q = compiled_model_q([input_image])[output_layer_q]
end = time.perf_counter()
time_ir = end - start
result_index_q = np.argmax(result_infer_q)
print("\nResult Index Quantized Model:", result_index_q)
print("Infer Time Quantized Model:", time_ir)



