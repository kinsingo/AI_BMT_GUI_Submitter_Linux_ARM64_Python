import os
import numpy as np
import cv2
from GUI_Mananger import ExecuteGUI, bmt
from PIL import Image
import torchvision.transforms.functional as F
from dx_engine import InferenceEngine

# Define the interface class for Classification using ONNX
class Classification_Implementation(bmt.AI_BMT_Interface):
    def __init__(self,use_customDataset=False):
        super().__init__()
        self.ie = None
        self.use_customDataset = use_customDataset

    def getOptionalData(self):
        optional = bmt.Optional_Data()
        optional.cpu_type = "DeepX M1 Python"
        optional.accelerator_type = "DeepX M1 Python"  
        optional.submitter = ""         
        optional.cpu_core_count = ""
        optional.cpu_ram_capacity = ""  # e.g., "32GB"
        optional.cooling = ""           # e.g., "Air"
        optional.cooling_option = ""    # e.g., "Active"
        optional.cpu_accelerator_interconnect_interface = ""  # e.g., "PCIe Gen5 x16"
        optional.benchmark_model = ""
        optional.operating_system = "Ubuntu24.04"
        return optional

    def getInterfaceType(self):
        if self.use_customDataset:
            return bmt.InterfaceType.ImageClassification_CustomDataset
        else:
            return bmt.InterfaceType.ImageClassification

    def initialize(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.ie = InferenceEngine(model_path)
        return True

    def preprocessVisionData(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply custom dataset preprocessing if needed
        if self.use_customDataset:
            image = Image.fromarray(image)
            image = F.resize(image, 232)
            image = F.center_crop(image, [224, 224])
            image = np.array(image)

        return [np.frombuffer(image, dtype=np.uint8)]

    def inferVision(self, preprocessed_data_list):
        output_tensors = []
        for _, preprocessed_data in enumerate(preprocessed_data_list):
            outputs = self.ie.run(preprocessed_data)
            output_tensors.append(outputs[0])
        return output_tensors
    
    def dataTransferVision(self, output_tensors):
        results = []
        for output_tensor in output_tensors:
            result = bmt.BMTVisionResult()
            result.classProbabilities = output_tensor.flatten()[:1000]
            results.append(result)
        return results
    

if __name__ == "__main__":
    interface = Classification_Implementation(use_customDataset=False)
    ExecuteGUI(interface)
    
    
