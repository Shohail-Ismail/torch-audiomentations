import torch
from torch import Tensor
from typing import Optional

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.object_dict import ObjectDict

class RMSNormalization(BaseWaveformTransform):
    
    """
    Adjusts gain so signal’s overall RMS matches `target_level_dbfs`

    Args:
      target_level_dbfs (float): desired RMS in dBFS.
      eps (float): small constant for numeric stability (default 1e-9)
    """
    
    supports_multichannel = True
    requires_sample_rate = False
    supports_target = False
    requires_target = False

    def __init__(self,
                 target_level_dbfs: float,
                 eps: float = 1e-9,
                 p: float = 1.0,
                 output_type: str = "dict"):
        # p = probability, output_type chooses dict vs tensor
        super().__init__(p = p, output_type = output_type)
        # Convert dBFS to linear gain
        self.target_amp = 10 ** (target_level_dbfs / 20.0)
        self.eps = eps

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        
        # Compute combined rms
        rms = samples.pow(2).mean(dim = (-1, -2), keepdim = True).sqrt()
        
        # Derive gain
        gain = self.target_amp / (rms + self.eps)
        
        # Apply gain
        out = samples * gain
        return ObjectDict(
            samples = out,
            sample_rate = sample_rate,
            targets = targets,
            target_rate = target_rate,
        )
