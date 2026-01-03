import json
from utils.model import load_lsnet_model
from utils.model_effnet import load_efficientnet_model
from utils.model_densenet import load_densenet_model
from utils.model_mobilenet import load_mobilenet_model
from utils.model_vit import load_vit_model
from utils.training_combined import count_parameters


NUM_CLASSES = 14
DEVICE = 'cuda'

counts = {}

lsnet_complexities = [ 't', 's', 'b' ]

for LSNET_COMPLEXITY in lsnet_complexities:
    model_key = f'lsnet_{LSNET_COMPLEXITY}'
    checkpoints_path = f'checkpoints_{model_key}/{model_key}_final.pth'
    model = load_lsnet_model(num_classes=NUM_CLASSES, device=DEVICE, checkpoint_path=checkpoints_path, model_complexity=LSNET_COMPLEXITY)
    
    total, trainable = count_parameters(model)
    counts[model_key] = { 'total_parameters': total, 'trainable_parameters': trainable }
    print(f"LSNet-{LSNET_COMPLEXITY}: ")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")


effnet_complexities = range(0, 8)

for EFFNET_COMPLEXITY in effnet_complexities:
    model_key = f'effnet_b{EFFNET_COMPLEXITY}'
    filename = f'effnetb{EFFNET_COMPLEXITY}'
    checkpoints_path = f'checkpoints_{model_key}/{filename}_final.pth'
    model = load_efficientnet_model(num_classes=NUM_CLASSES, device=DEVICE, checkpoint_path=checkpoints_path, model_complexity=EFFNET_COMPLEXITY)

    total, trainable = count_parameters(model)
    counts[model_key] = { 'total_parameters': total, 'trainable_parameters': trainable }
    print(f"EfficientNet-B{EFFNET_COMPLEXITY}: ")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")



model_key = f'densenet'
checkpoints_path = f'checkpoints_{model_key}/{model_key}_final.pth'
model = load_densenet_model(num_classes=NUM_CLASSES, device=DEVICE, checkpoint_path=checkpoints_path)
total, trainable = count_parameters(model)
counts[model_key] = { 'total_parameters': total, 'trainable_parameters': trainable }
print(f"DenseNet: ")
print(f"  Total parameters     : {total:,}")
print(f"  Trainable parameters : {trainable:,}")



model_key = f'vit'
filename = f'vit_best'
checkpoints_path = f'checkpoints_{model_key}/{filename}.pt'
model = load_densenet_model(num_classes=NUM_CLASSES, device=DEVICE, checkpoint_path=checkpoints_path)
total, trainable = count_parameters(model)
counts[model_key] = { 'total_parameters': total, 'trainable_parameters': trainable }
print(f"DenseNet: ")
print(f"  Total parameters     : {total:,}")
print(f"  Trainable parameters : {trainable:,}")

mobilenet_versions = [2]

for MOBILENET_VERSION in mobilenet_versions:
    model_key = f'mobilenet_v{MOBILENET_VERSION}'
    checkpoints_path = f'checkpoints_{model_key}/{model_key}_final.pth'
    model = load_mobilenet_model(num_classes=NUM_CLASSES, device=DEVICE, checkpoint_path=checkpoints_path, model_version=MOBILENET_VERSION)

    total, trainable = count_parameters(model)
    counts[model_key] = { 'total_parameters': total, 'trainable_parameters': trainable }
    print(f"MobileNetV{MOBILENET_VERSION}: ")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")


with open('parameter_counts.json', 'w') as f:
    json.dump(counts, f, indent=2)

