import os
import torch

MODEL_PRECISION = os.getenv("MODEL_PRECISION")
MODEL_REVISION = os.getenv("MODEL_REVISION")


if MODEL_PRECISION and not MODEL_REVISION:
    print("Warning: we no longer default to MODEL_REVISION=MODEL_PRECISION, please")
    print(f'explicitly set MODEL_REVISION="{MODEL_PRECISION}" if that\'s what you')
    print("want.")


def torch_dtype_from_precision(precision=MODEL_PRECISION):
    if precision == "fp16":
        return torch.float16
    return None