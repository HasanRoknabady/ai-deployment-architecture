# models/gender_preprocess/1/model.py

import numpy as np
import cv2
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        # Called once when the model is loaded
        pass

    def execute(self, requests):
        responses = []

        for request in requests:
            # دریافت batch: shape = [B, N]
            in_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_IMAGE")
            input_batch = in_tensor.as_numpy()  # np.ndarray of shape [B, N]

            batch_outputs = []

            for raw_bytes in input_batch:
                # Decode image from bytes
                img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)  # shape [H, W, 3] BGR
                if img is None:
                    raise ValueError("Failed to decode image.")

                # Resize to 224x224
                img = cv2.resize(img, (224, 224))

                # Convert BGR to RGB and normalize to [0, 1]
                img_rgb = img[:, :, ::-1].astype(np.float32) / 255.0

                # Convert to CHW format
                img_chw = np.transpose(img_rgb, (2, 0, 1))  # shape: [3, 224, 224]

                batch_outputs.append(img_chw)

            # Stack outputs → shape: [B, 3, 224, 224]
            output_array = np.stack(batch_outputs, axis=0).astype(np.float32)

            # Create output tensor
            output_tensor = pb_utils.Tensor("IMAGE_TENSOR", output_array)

            # Append response
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses
