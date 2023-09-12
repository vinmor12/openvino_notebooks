# FOURTH EXAMPLE
# This short tutorial shows how to convert a TensorFlow MobileNetV3
# image classification model to OpenVINO Intermediate Representation
# (OpenVINO IR) format, using Model Optimizer.

# IMPORTS ---------------------------------------------------------------------

print("Start Imports ")
import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from IPython.display import Markdown
from openvino.runtime import Core
print("Modules Have Been Imported ")

# SETTINGS --------------------------------------------------------------------

# The paths of the source and converted models.
print("Settings ...")
model_dir = Path("model")
model_dir.mkdir(exist_ok=True)
model_path = Path("model/v3-small_224_1.0_float")
ir_path = Path("model/v3-small_224_1.0_float.xml")

# DOWNLOAD TENSORFLOW MODEL ---------------------------------------------------

print("Start Download TensorFlow Model ...")
model = tf.keras.applications.MobileNetV3Small()
print("Download complete!")
print("Save TensorFlow Model ...")
model.save(model_path)

# CONVERT A MODEL TO OPENVINO IR ----------------------------------------------

# Construct the command for Model Optimizer.
mo_command = f"""mo
                 --saved_model_dir "{model_path}"
                 --input_shape "[1,224,224,3]"
                 --model_name "{model_path.name}"
                 --compress_to_fp16
                 --output_dir "{model_path.parent}"
                 """
mo_command = " ".join(mo_command.split())

# Run Model Optimizer if the IR model file does not exist
if not ir_path.exists():
    print("Exporting TensorFlow model to IR... This may take a few minutes.")
    os.system(mo_command)
else:
    print(f"IR model {ir_path} already exists.")

print("Operation Complete")
print("OpenVINO IR model is in 'model' dir!")

# Vincenzo  
    

