# THIRD EXAMPLE
# This example shows how to do inference with a object detection model.
# The horizontal-text-detection-0001 model from Open Model Zoo is used.

# IMPORT ----------------------------------------------------------------------
# Import libraries
print("Start Imports ")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
print("Modules Have Been Imported ")

# LOAD MODEL ------------------------------------------------------------------
print("\nWrite the Path of Model:\n(example: model/horizontal-text-detection-0001.xml)")
model_path = input()
# Initialize
ie = Core()
# Read OpenVINO IR model
model = ie.read_model(model=model_path)
# Compile OpenVINO IR model
print("\nWrite the target Device:\n(example: CPU)")
target_device = input()
compiled_model = ie.compile_model(model=model, device_name=target_device)
# Store input and output layers
input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output("boxes")

# LOAD IMAGE ------------------------------------------------------------------
# Text detection models expect an image in BGR format.
print("\nWrite the Path of Input Image:\n(example: ../data/image/intel_rnb.jpg)")
image_path = input()
image = cv2.imread(image_path)
# Define batch size, number of channels, height, width.
N, C, H, W = input_layer_ir.shape
# Resize the image to meet network expected input sizes.
resized_image = cv2.resize(image, (W, H))
# Reshape to the network input shape.
input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
# Plot image RGB
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));
print("\nClose the Image to Continue ...")
plt.show()

# INFERENCE -------------------------------------------------------------------
# Create an inference request.
boxes = compiled_model([input_image])[output_layer_ir]
# Remove zero only boxes.
boxes = boxes[~np.all(boxes == 0, axis=1)]
# Print boxes info
print("\nBoxes:\n",boxes)

# RESULT ----------------------------------------------------------------------
# For each detection, the description is in the [x_min, y_min, x_max, y_max, conf] format:
# The image passed here is in BGR format with changed width and height.
# To display it in colors expected by matplotlib, use cvtColor function
def convert_result_to_image(bgr_image, resized_image, boxes, threshold, conf_labels=True):
    # Define colors for boxes and descriptions.
    colors = {"red": (255, 0, 0), "green": (0, 255, 0)}
    # Fetch the image shapes to calculate a ratio.
    (real_y, real_x), (resized_y, resized_x) = bgr_image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
    # Convert the base image from BGR to RGB format.
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    # Iterate through non-zero boxes.
    for box in boxes:
        # Pick a confidence factor from the last place in an array.
        conf = box[-1]
        if conf > threshold:
            # Convert float to int and multiply corner position of each box by x and y ratio.
            # If the bounding box is found at the top of the image,
            # position the upper box bar little lower to make it visible on the image.
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y, 10)) if idx % 2
                else int(corner_position * ratio_x)
                for idx, corner_position in enumerate(box[:-1])
            ]
            # Draw a box based on the position, parameters in rectangle function are: 
            # image, start_point, end_point, color, thickness.
            rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 3)
            # Add text to the image based on position and confidence.
            # Parameters in text function are: image, text, bottom-left_corner_textfield, 
            # font, font_scale, color, thickness, line_type.
            if conf_labels:
                rgb_image = cv2.putText(
                    rgb_image,
                    f"{conf:.2f}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    colors["red"],
                    1,
                    cv2.LINE_AA,
                )
    return rgb_image
# Plot result
# Write threshold
print("\nWrite threshold:\nexample: 0.3")
threshold = float(input())
plt.figure(figsize=(10, 6))
plt.axis("off")
plt.imshow(convert_result_to_image(image, resized_image, boxes, threshold, conf_labels=True));
print("\nResult ...")
plt.show()

# Vincenzo
