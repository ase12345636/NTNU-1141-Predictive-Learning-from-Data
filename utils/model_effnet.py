from keras.layers import Input
from keras.applications.efficientnet import EfficientNetB0 as effnet
# 舊版
# from keras.layers.core import Dense
# 新版
from tensorflow.keras.layers import Dense
from keras.models import Model

import torch
from torchvision import models


def load_efficientnet_model(num_classes=14, device='cuda', checkpoint_path=None):
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")  # 預訓練模型
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    # input_shape=(224, 224, 3)
    # img_input = Input(shape=input_shape)
    # base_model = effnet(include_top=False, input_tensor=img_input, input_shape=input_shape, pooling="avg", weights='imagenet')

    # x = base_model.output
    # predictions = Dense(num_classes, activation="sigmoid", name="predictions")(x)
    # model = Model(inputs=img_input, outputs=predictions)

    state = None
    init_path = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        init_path = checkpoint_path
        print(f"Loaded checkpoint from {checkpoint_path}")

    # # Prefer a user-provided checkpoint if it exists; otherwise fall back to the hub weights
    # init_path = None
    # if checkpoint_path and os.path.exists(checkpoint_path):
    #     state = torch.load(checkpoint_path, map_location=device)
    #     init_path = checkpoint_path
    #     print(f"Loaded checkpoint from {checkpoint_path}")
    # else:
    #     ckpt_path = hf_hub_download("jameslahm/lsnet_t", "pytorch_model.bin")
    #     state = torch.load(ckpt_path, map_location=device, weights_only=True)
    #     init_path = ckpt_path
    #     print("Loaded pretrained weights from Hugging Face")

    # Only remove head weights if we're loading pretrained weights from HuggingFace
    # If loading from a checkpoint, keep all weights (including the trained head)
    if state:
        if not (checkpoint_path and os.path.exists(checkpoint_path)):
            state.pop("head.l.weight", None)
            state.pop("head.l.bias", None)
        model.load_state_dict(state, strict=False)

    model = model.to(device)
    model._checkpoint_path = init_path

    print("EfficientNet loaded OK!")
    return model

