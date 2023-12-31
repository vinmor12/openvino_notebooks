# Introduction to Detection in OpenVINO™ - "Hello Detection"
This notebook demonstrates how to do inference with detection model.

Notebook Contents
-
In this basic introduction to detection with OpenVINO, the [horizontal-text-detection-0001](https://docs.openvino.ai/latest/omz_models_model_horizontal_text_detection_0001.html) model from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/) is used. It detects text in images and returns blob of data in shape of [100, 5]. For each detection, a description is in the [x_min, y_min, x_max, y_max, conf] format.   

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F004-hello-detection%2F004-hello-detection.ipynb)

|                                                                                                                             |                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| <img src="https://user-images.githubusercontent.com/36741649/128489910-316aec49-4892-46f1-9e3c-b9d3646ef278.jpg" width=300> | <img src="https://user-images.githubusercontent.com/36741649/128489933-bf215a3f-06fa-4918-8833-cb0bf9fb1cc7.jpg" width=300> |

Script Usage
-
After cloning the repository, you can use this application in the following way:
+ inizialize OpenVINO environment (check the correct path of your "activate" file)
 ```
 openvino_env\Scripts\activate
 ```
+ switch to "004-hello--detection" directory
 ```
 cd openvino_notebooks\notebooks\004-hello--detection
 ```
+ run the python script
 ```
 python text_detection.py
 ```

