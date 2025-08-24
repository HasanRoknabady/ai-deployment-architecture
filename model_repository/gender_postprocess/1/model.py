# models/gender_postprocess/1/model.py
import numpy as np
import triton_python_backend_utils as pb_utils

# ------------- ثابت‌ها بر اساس مدل شما -------------
MALE_INDEX   = 1   # idx 1 -> Male
FEMALE_INDEX = 0   # idx 0 -> Female
# ---------------------------------------------------

class TritonPythonModel:
    def initialize(self, args):
        # هیچ وابستگی خارجی لازم نیست
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            logits = pb_utils.get_input_tensor_by_name(request, "LOGITS").as_numpy().astype(np.float32)
            # softmax
            exps  = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exps / np.sum(exps, axis=1, keepdims=True)

            male_probs   = probs[:, MALE_INDEX:MALE_INDEX+1]       # [B,1]
            female_probs = probs[:, FEMALE_INDEX:FEMALE_INDEX+1]   # [B,1]

            labels = np.array(
                [[b"Male"] if m > f else [b"Female"]
                 for m, f in zip(male_probs.squeeze(1), female_probs.squeeze(1))],
                dtype=object
            )

            outputs = [
                pb_utils.Tensor("MAN_PROB",   male_probs),
                pb_utils.Tensor("WOMAN_PROB", female_probs),
                pb_utils.Tensor("LABEL",      labels),
            ]
            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
        return responses
