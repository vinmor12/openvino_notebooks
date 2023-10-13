# TensorFlow to OpenVINO™ Model Conversion Tutorial
This tutorial explains how to convert [TensorFlow](https://www.tensorflow.org) models to OpenVINO IR. The notebook shows how to convert the [TensorFlow MobilenetV3 model](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) and then classify an image with OpenVINO Runtime.

Notebook Contents
-
The notebook uses [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) to convert the same [MobilenetV3](https://docs.openvino.ai/latest/omz_models_model_mobilenet_v3_small_1_0_224_tf.html) model used in the [001-hello-world notebook](../001-hello-world/001-hello-world.ipynb).

Script Usage
-
After cloning the repository, you can use this application in the following way:
+ inizialize OpenVINO environment (check the correct path of your "activate" file)
 ```
 openvino_env\Scripts\activate
 ```
+ switch to "101-tensorflow-to-openvino" directory
 ```
 cd openvino_notebooks\notebooks\101-tensorflow-to-openvino
 ```
+ run the python script
 ```
 python tf_to_ir.py
 ```
In this way, in "101-tensorflow-to-openvino" directory there will be a new "model" dir with IR "mobilenet-v3-small-1.0-224" model.
