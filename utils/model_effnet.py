import os
from keras.layers import Input
from keras.applications.efficientnet import EfficientNetB0 as effnet
from keras.models import Model

import torch
from torchvision import models
from huggingface_hub import hf_hub_download

def load_efficientnet_model(num_classes=14, device='cuda', checkpoint_path=None, model_complexity=0):
    if model_complexity == 7:
        model = models.efficientnet_b7(weights="IMAGENET1K_V1")  # 預訓練模型
    elif model_complexity == 6:
        model = models.efficientnet_b6(weights="IMAGENET1K_V1")  # 預訓練模型
    elif model_complexity == 5:
        model = models.efficientnet_b5(weights="IMAGENET1K_V1")  # 預訓練模型
    elif model_complexity == 4:
        model = models.efficientnet_b4(weights="IMAGENET1K_V1")  # 預訓練模型
    elif model_complexity == 3:
        model = models.efficientnet_b3(weights="IMAGENET1K_V1")  # 預訓練模型
    elif model_complexity == 2:
        model = models.efficientnet_b2(weights="IMAGENET1K_V1")  # 預訓練模型
    elif model_complexity == 1:
        model = models.efficientnet_b1(weights="IMAGENET1K_V1")  # 預訓練模型
    else:
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")  # 預訓練模型
    # model = models.efficientnet_b0(weights="chexpert")  # 預訓練模型
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    state = None
    init_path = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        init_path = checkpoint_path
        print(f"Loaded checkpoint from {checkpoint_path}")
    # else:
    #     ckpt_path = hf_hub_download("yashshinde0080/efficientnet-b0-finetuned-chest-xray-pneumonia", "pytorch_model.bin")
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

    print(f"EfficientNet B{model_complexity} loaded OK!")
    return model

def freeze_efficientnet_backbone(model):
    """
    Freeze all backbone parameters (no gradient update)
    """
    for param in model.features.parameters():
        param.requires_grad = False

def unfreeze_efficientnet_all(model):
    """
    Unfreeze all parameters
    """
    for param in model.classifier.parameters():
        param.requires_grad = True

