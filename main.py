import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torch.nn import functional as F

VARIANT = 10
EPOCHS = 10
BATCH_SIZE  = 64
LR = 1e-3
MOMENTUM = 0.9
DEVICE = torch.device('cpu')
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

def get_variant_pipelines(variant):
    pipelines = {
        1:  [("Scaling",     A.RandomScale(scale_limit=0.2, p=1.0)),
             ("Rotation",    A.Rotate(angle_range=(-15, 15), p=1.0))],
        2:  [("Cropping",    A.RandomCrop(24, 24, p=1.0)),
             ("Flipping",    A.HorizontalFlip(p=1.0))],
        3:  [("Translation", A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0,
                                                rotate_limit=0, p=1.0)),
             ("Brightness",  A.RandomBrightnessContrast(brightness_limit=0.3,
                                                        contrast_limit=0, p=1.0))],
        4:  [("Rotation",    A.Rotate(angle_range=(-15, 15), p=1.0)),
             ("Noise",       A.GaussNoise(p=1.0))],
        5:  [("Scaling",     A.RandomScale(scale_limit=0.2, p=1.0)),
             ("Cropping",    A.RandomCrop(24, 24, p=1.0))],
        6:  [("Brightness",  A.RandomBrightnessContrast(brightness_limit=0.3,
                                                        contrast_limit=0, p=1.0)),
             ("Contrast",    A.RandomBrightnessContrast(brightness_limit=0,
                                                        contrast_limit=0.3, p=1.0))],
        7:  [("Noise",       A.GaussNoise(p=1.0)),
             ("Blurring",    A.GaussianBlur(blur_limit=(3, 5), p=1.0))],
        8:  [("Rotation",    A.Rotate(angle_range=(-15, 15), p=1.0)),
             ("Translation", A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0,
                                                rotate_limit=0, p=1.0))],
        9:  [("Flipping",    A.HorizontalFlip(p=1.0)),
             ("Cropping",    A.RandomCrop(24, 24, p=1.0))],
        10: [("Contrast",    A.RandomBrightnessContrast(brightness_limit=0,
                                                        contrast_limit=0.3, p=1.0)),
             ("Noise",       A.GaussNoise(p=1.0))],
        11: [("Scaling",     A.RandomScale(scale_limit=0.2, p=1.0)),
             ("Flipping",    A.HorizontalFlip(p=1.0))],
    }
    return pipelines[variant]

def get_train_transforms():
    ops = [op for _, op in get_variant_pipelines(VARIANT)]
    ops += [A.Resize(32, 32),
            A.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2435,0.2616]), ##########
            ToTensorV2()]
    return A.Compose(ops)

val_transforms = A.Compose([
    A.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2435,0.2616]),
    ToTensorV2(),
])


class CIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.base = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]  # img is PIL
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

def show_augmentation_pipeline(variant):
    raw = datasets.CIFAR10('./data', train=True, download=False)
    img_pil, _ = raw[0]
    img_orig = np.array(img_pil)

    pipelines = get_variant_pipelines(variant)
    n = len(pipelines)

    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
    axes[0].imshow(img_orig, cmap='gray')
    axes[0].set_title('Original image')
    axes[0].axis('off')

    for i, (name, op) in enumerate(pipelines):
        aug = A.Compose([op])(image=img_orig)['image']
        if aug.shape != img_orig.shape:
            aug = A.Compose([A.Resize(32, 32)])(image=aug)['image']
        axes[i + 1].imshow(aug, cmap='gray')
        axes[i + 1].set_title(name)
        axes[i + 1].axis('off')

    plt.suptitle(f'Variant {variant}: preprocessing methods', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'augmentations_variant_{variant}.png', dpi=150)
    plt.close()
    print(f"SAVE: augmentations_variant_{variant}.png\n")

class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TokenMixer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_patches, expansion_factor, dropout)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_features, num_patches)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_features, expansion_factor, dropout)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor, dropout):
        super().__init__()  
        self.token_mixer = TokenMixer(
            num_features, num_patches, expansion_factor, dropout
        )
        self.channel_mixer = ChannelMixer(
            num_features, num_patches, expansion_factor, dropout
        )

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        # x.shape == (batch_size, num_patches, num_features)
        return x


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches


class MLPMixer(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_features=128,
        expansion_factor=2,
        num_layers=8,
        num_classes=10,
        dropout=0.5,
    ):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__()
        # per-patch fully-connected is equivalent to strided conv2d
        self.patcher = nn.Conv2d(
            in_channels, num_features, kernel_size=patch_size, stride=patch_size
        )
        self.mixers = nn.Sequential(
            *[
                MixerLayer(num_features, num_patches, expansion_factor, dropout)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_features, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_features)
        # patches.shape == (batch_size, num_patches, num_features)
        embedding = self.mixers(patches)
        # embedding.shape == (batch_size, num_patches, num_features)
        embedding = embedding.mean(dim=1)
        logits = self.classifier(embedding)
        return logits

def print_architecture(layers, activation):
    print("-" * 22)
    print("Architecture")
    print("-" * 22)
    act = activation if activation else "None"
    parts = [str(layers[0])]
    for size in layers[1:-1]:
        parts.append(f"{size} ({act})")
    parts.append(str(layers[-1]))
    print(" – ".join(parts))
    print()


def print_hyperparameters(lr, batch_size, epochs, extra=None):
    print("-" * 22)
    print("Hyperparameters")
    print("-" * 22)
    print(f"Learning Rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Max train epoch: {epochs}")
    if extra:
        for k, v in extra.items():
            print(f"{k}: {v}")
    print()


train_transforms = get_train_transforms()
train_full = CIFAR10('./data', train=True,  transform=train_transforms)
test_ds    = CIFAR10('./data', train=False, transform=val_transforms)

train_size = int(0.8 * len(train_full))
val_size   = len(train_full) - train_size
train_ds, val_ds = random_split(
    train_full, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}\n")

##show_augmentation_pipeline(VARIANT)


model = MLPMixer()
params = list(model.parameters())
total_params = sum(p.numel() for p in params)

# Лог архитектуры и гиперпараметров
print_hyperparameters(
    lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS,
    extra={
        'Optimizer': 'SGD',
        'Momentum': MOMENTUM,
        'Loss': 'CrossEntropyLoss',
        'Total parameters': f"{total_params:,} (~{total_params/1e6:.2f}M)",
        'Data type': 'float32',
    }
)

optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM)
criterion = nn.CrossEntropyLoss()

@torch.no_grad()
def evaluate(loader):
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(DEVICE).to(torch.float32)
        y = y.to(DEVICE)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model_loss_sum = 0.0
    train_correct, train_total = 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch #{epoch}", leave=False)
    for x, y in pbar:
        x = x.to(DEVICE).to(torch.float32)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        model_loss_sum += loss.item() * x.size(0)
        train_correct += (out.argmax(1) == y).sum().item()
        train_total += y.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = model_loss_sum / len(train_ds)
    train_acc = train_correct / train_total

    val_loss, val_acc = evaluate(val_loader)
    elapsed = time.time() - t0

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    best_val_acc = max(best_val_acc, val_acc)

    print(f"Epoch #{epoch}: train_accuracy= {train_acc:.4f} ; "
          f"test_accuracy= {val_acc:.4f} ; time= {elapsed:.2f}s")


print("\n" + "-" * 22)
print("Final evaluation on test set")
print("-" * 22)
test_loss, test_acc = evaluate(test_loader)
print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")
print(f"Best val accuracy: {best_val_acc:.4f}\n")

checkpoint = {
    'variant': VARIANT,
    'architecture': 'MLP-Mixer',
    'state_dict': model.state_dict(),   # ← вот так
    'test_acc': test_acc,
}
torch.save(checkpoint, 'mlp_mixer_weights.pt')
print("Веса сохранены: mnist_mlp_weights.pt")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history['train_loss'], label='train')
axes[0].plot(history['val_loss'], label='val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)

axes[1].plot(history['val_acc'], color='green', label='val_acc')
axes[1].axhline(0.99, color='red', linestyle='--', label='0.99')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Validation accuracy')
axes[1].legend(); axes[1].grid(True)

## plt.suptitle(f'Variant {VARIANT} — MLP ~1M params, SGD')
plt.tight_layout()
plt.savefig(f'training_curves_variant_{VARIANT}.png')
plt.close()
print(f"График сохранён: training_curves_variant_{VARIANT}.png")