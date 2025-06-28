import torch
from torch import Tensor
from typing import Optional

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.object_dict import ObjectDict
from ..utils import db_to_amplitude

class RMSNormalization(BaseWaveformTransform):
    
    """
    Adjusts gain so signal’s overall RMS matches `target_level_dbfs`
    """
    
    supports_multichannel = True
    requires_sample_rate = False
    supports_target = False
    requires_target = False

    def __init__(
        self,
        target_level_dbfs: float,
        eps: float = 1e-9,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: Optional[str] = None,
        sample_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
        output_type: Optional[str] = None,
    ):
        """
        Args:
          target_level_dbfs (float): Desired RMS in dBFS.
          eps (float): Small constant for numeric stability.
          mode (str): “per_example”, “per_channel” or “per_batch”.
          p (float): Probability of applying the transform (default 0.5).
          p_mode (Optional[str]): Secondary probability mode.
          sample_rate (Optional[int]): Sample rate, if required.
          target_rate (Optional[int]): Target sample rate.
          output_type (Optional[str]): “tensor” or “dict” output.
        """
        super().__init__(
            mode = mode,
            p = p,
            p_mode = p_mode,
            sample_rate = sample_rate,
            target_rate = target_rate,
            output_type = output_type,
        )
        
        # Convert target dBFS to linear amplitude
        self.target_amp = db_to_amplitude(target_level_dbfs)
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
