# READING, COMPILING AND INFERENCE OF THE MODEL WITH TRANSFER LEARNING

# IMPORTS ---------------------------------------------------------------------

print("Start Imports ")
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from openvino.runtime import Core
print("Modules Have Been Imported")

# CONVERT MODEL ---------------------------------------------------------------

# Define path
model_path = Path("model/tl_model")
ir_path = Path("model")

# Model Conversion with Model Optimizer
mo_command = f"""mo
                 --saved_model_dir "{model_path}"
                 --input_shape "[1,224,224,3]"
                 --model_name "{model_path.name}"
                 --compress_to_fp16
                 --output_dir "{model_path.parent}"
                 """
mo_command = " ".join(mo_command.split())

os.system(mo_command)


# READ AND COMPILE MODEL ------------------------------------------------------

# Initialize OpenVINO Runtime
ie = Core()

# Read model 
model = ie.read_model("model/tl_model.xml")

# Compile model
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_key = compiled_model.input(0)
output_key = compiled_model.output(0)
network_input_shape = input_key.shape

# LOAD INPUT IMAGES -----------------------------------------------------------
foto = ["foto1.jpg", "foto2.jpg", "foto3.jpg", "foto4.jpg", "foto5.jpg", "foto6.jpg", "foto7.jpg", "foto8.jpg", "foto9.jpg"]
nfoto = 9;
image = list(range(nfoto))
resized_image = list(range(nfoto))
input_image = list(range(nfoto))

for i in list(range(nfoto)):
    # Load image
    print("Load image ... ")
    # The MobileNet network expects images in RGB format.
    image[i] = cv2.cvtColor(cv2.imread(filename="foto/"+foto[i]), code=cv2.COLOR_BGR2RGB)
    # Resize the image to the network input shape.
    print("Pre-processing image ... ")
    resized_image[i] = cv2.resize(src=image[i], dsize=(224, 224))
    # Transpose the image to the network input shape.
    input_image[i] = np.expand_dims(resized_image[i], 0)
    print("Show image ... ")
    plt.subplot(3,3,i+1)
    plt.imshow(image[i]);

# INFERENCE -------------------------------------------------------------------
print("Inference ... ")
print("Detected class:  ")

result = list(range(nfoto))
class_names = open("class_names.txt").read().splitlines()

def threshold (n):
    r = n
    f = 0
    for i in n:
        if i>=0:
            r[f]=1
        else:
            r[f]=0
        f=f+1
    return r

for i in list(range(nfoto)):
    result[i] = compiled_model([input_image[i]])[output_key]
    result[i] = threshold(result[i])
    result[i] = result[i].astype("i")
    #print(result[i])
    for i in result[i]:
        print(class_names[i[0]])

plt.show()

# Vincenzo



