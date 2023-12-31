# Introduction to Segmentation in OpenVINO - "Hello Segmentation" Example
This notebook demonstrates how to do inference with segmentation model.   

Notebook Contents
-
A very basic introduction to segmentation with OpenVINO. This notebook uses the [road-segmentation-adas-0001](https://docs.openvino.ai/latest/omz_models_model_road_segmentation_adas_0001.html) model from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/) and an input image downloaded from [Mapillary Vistas](https://www.mapillary.com/dataset/vistas). ADAS stands for Advanced Driver Assistance Services. The model recognizes four classes: background, road, curb and mark.   

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F003-hello-segmentation%2F003-hello-segmentation.ipynb)

|                                                                                                                             |                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| <img src="https://user-images.githubusercontent.com/36741649/127848003-9e45c8da-2e43-48ac-803f-9f51a8e9ea89.jpg" width=300> | <img src="https://user-images.githubusercontent.com/36741649/127847882-6305d483-f2ce-4c2f-a3b5-8573d1522d15.png" width=300> |

Script Usage
-
After cloning the repository, you can use this application in the following way:
+ inizialize OpenVINO environment (check the correct path of your "activate" file)
 ```
 openvino_env\Scripts\activate
 ```
+ switch to "003-hello-segmentation" folder
 ```
 cd openvino_notebooks\notebooks\003-hello-segmentation
 ```
+ run the python script
 ```
 python segmentation.py
 ```
