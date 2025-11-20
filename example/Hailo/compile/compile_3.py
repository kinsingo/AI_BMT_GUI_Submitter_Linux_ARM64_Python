from hailo_sdk_client import ClientRunner
import os

os.makedirs("ObjectDetection_compiled_hars", exist_ok=True)

for quantized_model_har_name in os.listdir("ObjectDetection_quantized_hars"):
    if not quantized_model_har_name.endswith(".har"):
        continue

    har_path = os.path.join("ObjectDetection_quantized_hars", quantized_model_har_name)
    print(f"[INFO] Compiling {har_path} ...")

    try:
        runner = ClientRunner(har=har_path)
        hef = runner.compile()
    except Exception as e:
        print(f"[ERROR] Failed to compile {quantized_model_har_name}: {e}")
        # 여기서 로그만 남기고 다음 모델로 넘어감
        continue

    base_name = os.path.splitext(quantized_model_har_name)[0]
    out_path = f"ObjectDetection_compiled_hars/{base_name}_compiled.hef"
    with open(out_path, "wb") as f:
        f.write(hef)

    print(f"[SUCCESS] Saved {out_path}")
