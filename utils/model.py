import os
import torch
import lsnet.model.lsnet as lsnet
from huggingface_hub import hf_hub_download


def load_lsnet_model(num_classes=14, device='cuda', checkpoint_path=None, model_complexity='t'):
    """Load LSNet weights from a local checkpoint when provided, otherwise from Hugging Face."""
    if model_complexity == 'b':
        model = lsnet.lsnet_b(
            num_classes=num_classes,
            distillation=False,
            pretrained=False,
            frozen_stages=0
        )
    elif model_complexity == 's':
        model = lsnet.lsnet_s(
            num_classes=num_classes,
            distillation=False,
            pretrained=False,
            frozen_stages=0
        )
    else:
        model_complexity = 't'
        model = lsnet.lsnet_t(
            num_classes=num_classes,
            distillation=False,
            pretrained=False,
            frozen_stages=0
        )

    # Prefer a user-provided checkpoint if it exists; otherwise fall back to the hub weights
    init_path = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        init_path = checkpoint_path
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        ckpt_path = hf_hub_download(f"jameslahm/lsnet_{model_complexity}", "pytorch_model.bin")
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        init_path = ckpt_path
        print("Loaded pretrained weights from Hugging Face")

    # Only remove head weights if we're loading pretrained weights from HuggingFace
    # If loading from a checkpoint, keep all weights (including the trained head)
    if not (checkpoint_path and os.path.exists(checkpoint_path)):
        keys_to_remove = [k for k in state.keys() if k.startswith("head.")]
        for k in keys_to_remove:
            print(f'Remove: {k}')
            state.pop(k, None)
        # state.pop("head.l.weight", None)
        # state.pop("head.l.bias", None)

    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model._checkpoint_path = init_path
    print("LSNet loaded OK!")
    return model

def freeze_lsnet_backbone(model):
    """
    Freeze all backbone parameters (no gradient update)
    """
    for name, param in model.named_parameters():
        if not name.startswith("head."):
            param.requires_grad = False

def unfreeze_lsnet_all(model):
    """
    Unfreeze all parameters
    """
    for param in model.parameters():
        param.requires_grad = True
