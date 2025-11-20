import os
import numpy as np
import cv2

from GUI_Mananger import ExecuteGUI, bmt
from hailo_platform import (
    HEF, VDevice, FormatType, HailoSchedulingAlgorithm
)
from PIL import Image
import torchvision.transforms.functional as F

WIDTH = 224
HEIGHT = 224

class Classification_Implementation(bmt.AI_BMT_Interface):
    def __init__(self, use_customDataset=False):
        super().__init__()
        # Hailo objects
        self.vdevice = None
        self.infer_model = None
        self.config_ctx = None
        self.configured_model = None

        # I/O names
        self.input_name = None
        self.output_name = None

        # Reusable buffers/binding
        self._out_shape = None
        self._out_buf = None
        self._binding = None
        self._ready = False  # async ready 호출을 1회만 하기 위함
        self.use_customDataset = use_customDataset

    def getOptionalData(self):
        optional = bmt.Optional_Data()
        optional.cpu_type = "Broadcom BCM2712 quad-core Arm Cortex A76 @ 2.4GHz"
        optional.accelerator_type = "Hailo-8" 
        optional.submitter = "Hailo(Custom Dataset)" if self.use_customDataset else "Hailo"
        optional.cpu_core_count = "4"
        optional.cpu_ram_capacity = "8GB"
        optional.cooling = "Air"
        optional.cooling_option = "Active"
        optional.cpu_accelerator_interconnect_interface = "PCIe 3.0 x4"
        optional.benchmark_model = "mobilenet_v2_opset10"
        optional.operating_system = "Ubuntu 24.04.2 LTS"
        return optional

    def getInterfaceType(self):
        if self.use_customDataset:
            return bmt.InterfaceType.ImageClassification_CustomDataset
        else:
            return bmt.InterfaceType.ImageClassification

    def initialize(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # 1) VDevice (single-stream)
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "BMT_SINGLE"
        self.vdevice = VDevice(params)

        # 2) HEF & infer model
        hef = HEF(model_path)
        self.infer_model = self.vdevice.create_infer_model(model_path)
        self.infer_model.set_batch_size(1)

        # 3) I/O format (C++ 의도대로)
        self.infer_model.input().set_format_type(FormatType.UINT8)
        for out in self.infer_model.outputs:
            self.infer_model.output(out.name).set_format_type(FormatType.FLOAT32)

        # 4) I/O names
        self.input_name = hef.get_input_vstream_infos()[0].name
        self.output_name = hef.get_output_vstream_infos()[0].name

        # 5) Configure once
        self.config_ctx = self.infer_model.configure()
        self.configured_model = self.config_ctx.__enter__()
        self.configured_model.set_scheduler_priority(0)

        # 6) **Create reusable output buffer & binding (once)**
        self._out_shape = self.infer_model.output(self.output_name).shape
        self._out_buf = { self.output_name: np.empty(self._out_shape, dtype=np.float32) }
        self._binding = self.configured_model.create_bindings(output_buffers=self._out_buf)

        # 7) **Call ready once**
        self.configured_model.wait_for_async_ready(timeout_ms=10000)
        self._ready = True

        return True

    def preprocessVisionData(self, image_path: str):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.use_customDataset:
            img = Image.fromarray(img)
            img = F.resize(img, 232)
            img = F.center_crop(img, [224, 224])
            img = np.array(img)

        return np.ascontiguousarray(img, dtype=np.uint8)

    def inferVision(self, preprocessed_data_list):
        outputs = []
        for img in preprocessed_data_list:
            self._binding.input().set_buffer(img)
            self.configured_model.run([self._binding], timeout=10000)
            out = self._binding.output(self.output_name).get_buffer()
            outputs.append(out.reshape(-1))
        return outputs

    def dataTransferVision(self, output_tensors):
        results = []
        for out in output_tensors:
            r = bmt.BMTVisionResult()
            r.classProbabilities = out
            results.append(r)
        return results

if __name__ == "__main__":
    use_customDataset=False
    print("Hailo(Custom Dataset)" if use_customDataset else "Hailo(Normal Dataset)")
    interface = Classification_Implementation(use_customDataset)
    ExecuteGUI(interface)
