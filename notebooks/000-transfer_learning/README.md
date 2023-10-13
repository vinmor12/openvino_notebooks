# Introduction to Transfer Learning
In this tutorial, you will learn how to classify images of cats and dogs by using transfer learning with API Keras from a pre-trained network.

You will use a dataset containing several thousand images of cats and dogs.

Finally, you will do inference with OpenVINO.


Scripts Usage
-
After cloning the repository, you can use these scripts in the following way:
+ inizialize OpenVINO environment (check the correct path of your "activate" file)
 ```
 openvino_env\Scripts\activate
 ```
+ switch to "000-transfer_learning" folder
 ```
 cd openvino_notebooks\notebooks\000-transfer_learning
 ```
+ run the python script to do transfer learning
 ```
 python transfer_learning.py
 ```
+ run the python script to do inference with OpenVINO
 ```
 python conversion_and_inference.py
 ```
