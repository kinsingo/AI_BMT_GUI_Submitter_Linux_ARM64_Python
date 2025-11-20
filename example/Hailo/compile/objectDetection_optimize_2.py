
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.eager.context import eager_mode

# import the hailo sdk client relevant classes
from hailo_sdk_client import ClientRunner, InferenceContext

images_path = "ObjectDetection_Calibration_Images"
images_list = [img_name for img_name in os.listdir(images_path) if os.path.splitext(img_name)[1] == ".jpg"]
calib_dataset = np.zeros((len(images_list), 640, 640, 3))
for idx, img_name in enumerate(sorted(images_list)):
    img = np.array(Image.open(os.path.join(images_path, img_name)))
    assert img.shape == (640, 640, 3), f"{img_name} has unexpected shape {img.shape}"
    calib_dataset[idx] = img

# Create ObjectDetection_quantized_hars folder if it doesn't exist
os.makedirs("ObjectDetection_quantized_hars", exist_ok=True)

# Get all HAR files from ObjectDetection_hars folder
har_files = [f for f in os.listdir("ObjectDetection_hars") if f.endswith(".har")]

for har_file in har_files:
    model_name = har_file.replace("_hailo_model.har", "")
    hailo_model_har_name = f"ObjectDetection_hars/{har_file}"
    
    print(f"[INFO] Processing {model_name}...")
    
    try:

        quantized_model_har_path = f"ObjectDetection_quantized_hars/{model_name}_bgr2rgb_normalized_quantized_model.har"
        if os.path.exists(quantized_model_har_path):
            print(f"[INFO]{quantized_model_har_path} is existing.. continue..")
            continue

        runner = ClientRunner(har=hailo_model_har_name)
        # Load the model script to ClientRunner so it will be considered on optimization
        alls = (
            "model_optimization_flavor(optimization_level=2,compression_level=2,batch_size=2)\n"
            "color_convert = input_conversion(bgr_to_rgb, emulator_support=True)\n"
            "norm_layer = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"
        )
        runner.load_model_script(alls)
        # Call Optimize to perform the optimization process
        runner.optimize(calib_dataset) 
        # Save the result state to a Quantized HAR file
        runner.save_har(quantized_model_har_path)
        print(f"[SUCCESS] {model_name} optimized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to optimize {model_name}: {str(e)}")
        continue

# Create ObjectDetection_quantized_hars folder if