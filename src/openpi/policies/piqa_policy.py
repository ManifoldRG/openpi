import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class PiqaInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0
        inputs = {}

        # Pass the prompt (aka language instruction) to the model.
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        
        # Create inputs dict. 
        dummy_image = _parse_image(np.zeros((224, 224, 3), dtype=np.uint8))
        inputs = {
            "prompt": data.get("prompt", ""),
            "image": {
                # rename these to base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb if using pi0-FAST
                "base_0_rgb": dummy_image,
                "left_wrist_0_rgb": np.zeros_like(dummy_image),
                "right_wrist_0_rgb": np.zeros_like(dummy_image),
            },
            "image_mask": {
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "base_0_rgb": np.False_ if mask_padding else np.True_,
                "left_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        return inputs