# Introduction to OpenVINO - "Hello World" Example
This notebook shows how to do inference on an OpenVINO IR model.

Notebook Contents
-
This notebook demonstrates usage of [MobileNet V3](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/mobilenet-v3-small-1.0-224-tf/README.md) from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/)   

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F001-hello-world%2F001-hello-world.ipynb)

![classification](https://user-images.githubusercontent.com/36741649/127172572-1cdab941-df5f-42e2-a367-2b334a3db6d8.jpg)

Script Usage
-
After cloning the repository, you can use this application in the following way:
+ inizialize OpenVINO environment (check the correct path of your "activate" file)
 ```
 openvino_env\Scripts\activate
 ```
+ switch to "001-hello-world" folder
 ```
 cd openvino_notebooks\notebooks\001-hello-world
 ```
+ run the python script
 ```
 python image_classification.py
 ```
