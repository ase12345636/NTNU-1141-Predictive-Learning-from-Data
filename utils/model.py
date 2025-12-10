import torch
import lsnet.model.lsnet as lsnet
from huggingface_hub import hf_hub_download

def load_lsnet_model(num_classes=14, device='cuda'):
    """Load pre-trained LSNet model"""
    ckpt_path = hf_hub_download("jameslahm/lsnet_t", "pytorch_model.bin")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state.pop("head.l.weight", None)
    state.pop("head.l.bias", None)
    
    model = lsnet.lsnet_t(
        num_classes=num_classes,
        distillation=False,
        pretrained=False,
        frozen_stages=0
    )
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    print("LSNet loaded OK!")
    return model
