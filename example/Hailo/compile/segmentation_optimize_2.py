
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.eager.context import eager_mode

# import the hailo sdk client relevant classes
from hailo_sdk_client import ClientRunner, InferenceContext

images_path = "Segmentation_Calibration_Images"
images_list = [img_name for img_name in os.listdir(images_path) if os.path.splitext(img_name)[1] == ".jpg"]
calib_dataset = np.zeros((len(images_list), 520, 520, 3))
for idx, img_name in enumerate(sorted(images_list)):
    img = np.array(Image.open(os.path.join(images_path, img_name)))
    assert img.shape == (520, 520, 3), f"{img_name} has unexpected shape {img.shape}"
    calib_dataset[idx] = img

# deeplabv3_mobilenet_v3_large_opset12 : X (실패)
# deeplabv3_resnet50_opset12 : O (성공)
# deeplabv3_resnet101_opset12 : O (성공)
# fcn_resnet50_opset12 : O (성공)
# fcn_resnet101_opset12 : O (성공)
for model_name in ["deeplabv3_resnet50_opset12", "deeplabv3_resnet101_opset12", "fcn_resnet50_opset12", "fcn_resnet101_opset12"]:
    hailo_model_har_name = f"{model_name}_hailo_model.har"
    assert os.path.isfile(hailo_model_har_name), "Please provide valid path for HAR file"
    runner = ClientRunner(har=hailo_model_har_name)

    # Load the model script to ClientRunner so it will be considered on optimization
    alls = (
            "model_optimization_flavor(optimization_level=2,compression_level=2,batch_size=2)\n"
            "normalization1=normalization([123.675,116.28,103.53],[58.395,57.12,57.375])\n"
        )
    
    runner.load_model_script(alls)

    # Call Optimize to perform the optimization process
    runner.optimize(calib_dataset)

    # Save the result state to a Quantized HAR file
    quantized_model_har_path = f"{model_name}_normalized_quantized_model.har"
    runner.save_har(quantized_model_har_path)