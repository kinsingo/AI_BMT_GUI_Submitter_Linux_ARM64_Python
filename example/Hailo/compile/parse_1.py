# import the ClientRunner class from the hailo_sdk_client package
from hailo_sdk_client import ClientRunner

# Segmentation
# for onnx_model_name in ["deeplabv3_mobilenet_v3_large_opset12", "deeplabv3_resnet50_opset12", "deeplabv3_resnet101_opset12", "fcn_resnet50_opset12", "fcn_resnet101_opset12"]:
#     onnx_path = f"Segmentation_onnxs/{onnx_model_name}.onnx"
#     runner = ClientRunner(hw_arch="hailo8")
#     hn, npz = runner.translate_onnx_model(
#         onnx_path,
#         onnx_model_name,
#         start_node_names=["input"],
#         end_node_names=["output"],   
#         net_input_shapes={"input": [1, 3, 520, 520]},
#     )
#     hailo_model_har_name = f"{onnx_model_name}_hailo_model.har"
#     runner.save_har(hailo_model_har_name)

# Object Detection 
import os

# Create ObjectDetection_hars folder if it doesn't exist
os.makedirs("ObjectDetection_hars", exist_ok=True)

onnx_files = [f.replace(".onnx", "") for f in os.listdir("ObjectDetection_onnxs") if f.endswith(".onnx")]
runner = ClientRunner(hw_arch="hailo8")

# Define end node names for different model types
end_nodes_config = {
    "yolov5": [
        "/model.24/Sigmoid_1",
        "/model.24/Sigmoid",
        "/model.24/Sigmoid_2",
    ], 
    #"yolov5": ["/model.24/Concat_3"],  # YOLOv5nu models
    "yolov8": ["/model.22/Concat_3"],  # YOLOv8 models (may need adjustment)
    "yolov9": ["/model.22/Concat_3"],  # YOLOv9 models (may need adjustment)
}

for onnx_model_name in onnx_files:
    onnx_path = f"ObjectDetection_onnxs/{onnx_model_name}.onnx"
    
    # Determine end nodes based on model name
    end_nodes = None
    for model_type, nodes in end_nodes_config.items():
        if model_type in onnx_model_name.lower():
            end_nodes = nodes
            break
    
    try:
        if end_nodes:
            hn, npz = runner.translate_onnx_model(
                onnx_path,
                onnx_model_name,
                end_node_names=end_nodes,
                net_input_shapes={"images": [1, 3, 640, 640]},
            )
        else:
            hn, npz = runner.translate_onnx_model(
                onnx_path,
                onnx_model_name,
                net_input_shapes={"images": [1, 3, 640, 640]},
            )
        hailo_model_har_name = f"ObjectDetection_hars/{onnx_model_name}_hailo_model.har"
        runner.save_har(hailo_model_har_name)
        print(f"[SUCCESS] {onnx_model_name} converted successfully")
    except Exception as e:
        print(f"[ERROR] Failed to convert {onnx_model_name}: {str(e)}")
        # Extract recommended end node from error message if available
        if "Please try to parse the model again, using these end node names:" in str(e):
            recommended_node = str(e).split("using these end node names:")[-1].strip()
            print(f"[RECOMMENDATION] Try using end_node_names=['{recommended_node}']")
        continue

