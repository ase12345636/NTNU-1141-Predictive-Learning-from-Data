import os
import torch
import timm
from huggingface_hub import hf_hub_download


# model = create_model(args.model, pretrained= True)

def load_vit_model(num_classes=14, device='cuda', checkpoint_path=None):
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,    # ImageNet
        num_classes=num_classes
    )

    state = None
    init_path = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        init_path = checkpoint_path
        print(f"Loaded checkpoint from {checkpoint_path}")
    # else:
    #     ckpt_path = hf_hub_download("google/vit-base-patch16-224-in21k", "pytorch_model.bin")
    #     state = torch.load(ckpt_path, map_location=device, weights_only=True)
    #     init_path = ckpt_path
    #     print("Loaded pretrained weights from Hugging Face")

    # Only remove head weights if we're loading pretrained weights from HuggingFace
    # If loading from a checkpoint, keep all weights (including the trained head)
    if state:
        if not (checkpoint_path and os.path.exists(checkpoint_path)):
            # state.pop("head.l.weight", None)
            # state.pop("head.l.bias", None)
            keys_to_remove = [k for k in state.keys() if k.startswith("head.")]
            for k in keys_to_remove:
                print(f'Remove: {k}')
                state.pop(k, None)
        model.load_state_dict(state, strict=False)

    model = model.to(device)
    model._checkpoint_path = init_path

    print("ViT loaded OK!")
    return model

