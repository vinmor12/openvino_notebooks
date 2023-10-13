# Post-Training Optimization
This example shows how to optimize an image classification model with POT ("Post-Training Optimization Tool").
A pre-trained MobileNetV3 model from Open Model Zoo is used in this example.

Finally, you will do inference with OpenVINO.


Script Usage
-
After cloning the repository, you can use this application in the following way:
+ inizialize OpenVINO environment (check the correct path of your "activate" file)
 ```
 openvino_env\Scripts\activate
 ```
+ switch to "000-quantize-model" folder
 ```
 cd openvino_notebooks\notebooks\000-quantize-model
 ```
+ unzip "dataset.zip" file
+ run the python script
 ```
 python inference_quantized_model.py
 ```
