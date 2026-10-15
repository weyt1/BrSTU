import argparse
import random
import time
from pathlib import Path

# import mlflow
# import mlflow.pytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugmentPolicy
from tqdm import tqdm

SEED = 3407
VARIANT = 10
EPOCHS = 200
BATCH_SIZE = 128
LR = 1e-3
WARMUP_EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2470, 0.2435, 0.2616]


def seed_everything(seed=SEED):

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CIFAR10Dataset(Dataset):

    def __init__(self, root, train, transform):
        self.base = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        return self.transform(image), label


class MLP(nn.Module):

    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim),
                                 nn.Dropout(dropout))
        
    def forward(self, x): return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, tokens, channels, token_hidden=64, channel_hidden=512, dropout=0.0):
        super().__init__()
        self.norm_token = nn.LayerNorm(channels)
        self.token_mlp = MLP(tokens, token_hidden, dropout)
        self.norm_channel = nn.LayerNorm(channels)
        self.channel_mlp = MLP(channels, channel_hidden, dropout)

    def forward(self, x):
        x = x + self.token_mlp(self.norm_token(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mlp(self.norm_channel(x))
        return x


class MLPMixer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, dim=128, depth=8,
                 token_hidden=64, channel_hidden=512, classes=10):
        super().__init__()
        assert image_size % patch_size == 0
        side = image_size // patch_size
        tokens = side * side
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        self.blocks = nn.Sequential(*[
            MixerBlock(tokens, dim, token_hidden, channel_hidden)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = self.blocks(x)
        return self.head(self.norm(x).mean(dim=1))


def apply_cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    _, _, h, w = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_ratio), int(h * cut_ratio)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
    return mixed, y, y[index], lam


def train_transforms():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.20), value=0),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transforms():
    return transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)
    ])

def evaluate(model, loader, criterion, amp_enabled):
    model.eval(); loss_sum = correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_enabled):
                out = model(x); loss = criterion(out, y)
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
    return loss_sum / total, correct / total

def main(args):
    seed_everything(args.seed)

    #mlflow.set_experiment(args.experiment)

    amp_enabled = DEVICE.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled) ## для работы с float16, чтобы градиентв не округлялись до нуля

    root = Path(args.data)

    train_aug = CIFAR10Dataset(root, True, train_transforms())
    train_clean = CIFAR10Dataset(root, True, eval_transforms())
    test_ds = CIFAR10Dataset(root, False, eval_transforms())
    indices = np.arange(len(train_aug)); np.random.default_rng(args.seed).shuffle(indices)
    split = int(0.8 * len(indices))
    train_ds = Subset(train_aug, indices[:split])
    val_ds = Subset(train_clean, indices[split:])


    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)


    model = MLPMixer().to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=5e-5)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, args.warmup)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs-args.warmup), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [args.warmup])
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}; best_val_acc = 0.0

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Device: {DEVICE} | parameters: {total_params:,}')

    with mlflow.start_run(run_name=f'Variant_{args.variant}_MixerNano'):
        mlflow.log_params(vars(args) | {'device': str(DEVICE), 'parameters': total_params})
    #перед раскомментирование сделать табуляцию
        for epoch in range(1, args.epochs + 1):

            start = time.time(); model.train(); loss_sum = 0.0; correct = total = 0

            for x, y in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                use_cutmix = np.random.rand() < args.cutmix_prob

                if use_cutmix: x, ya, yb, lam = apply_cutmix(x, y)

                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_enabled):
                    out = model(x)
                    loss = (lam * criterion(out, ya) + (1-lam) * criterion(out, yb)) if use_cutmix else criterion(out, y)

                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                loss_sum += loss.item() * y.size(0); pred = out.argmax(1)
                correct += ((lam * (pred == ya).float() + (1-lam) * (pred == yb).float()).sum().item() if use_cutmix else (pred == y).sum().item())
                total += y.size(0)

        scheduler.step()

        train_loss, train_acc = loss_sum / total, correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, amp_enabled)

        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)

        best_val_acc = max(best_val_acc, val_acc)

        mlflow.log_metrics({'train_loss': train_loss, 'train_accuracy': train_acc, 'val_loss': val_loss, 'val_accuracy': val_acc, 'lr': optimizer.param_groups[0]['lr'], 'epoch_time_sec': time.time()-start}, step=epoch)

        print(f'Epoch {epoch:03d}: train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | time={time.time()-start:.1f}s')

        print('\n' + '-' * 22); print('Final evaluation on test set'); print('-' * 22)

        test_loss, test_acc = evaluate(model, test_loader, criterion, amp_enabled)

        print(f'Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}')
        print(f'Best val accuracy: {best_val_acc:.4f}\n')

        mlflow.log_metric('final_test_accuracy', test_acc); mlflow.log_metric('final_test_loss', test_loss)

        model.eval(); example_input = torch.randn(1, 3, 32, 32, device=DEVICE)

        mlflow.pytorch.log_model(model, 'model', input_example=example_input, serialization_format='pickle')

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history['train_loss'], label='train'); axes[0].plot(history['val_loss'], label='val'); axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
        axes[1].plot(history['val_acc'], color='green', label='val_acc'); axes[1].set_title('Validation accuracy'); axes[1].legend(); axes[1].grid(True)
        plt.tight_layout(); chart_path = f'training_curves_variant_{args.variant}.png'; plt.savefig(chart_path, dpi=150); plt.close(); mlflow.log_artifact(chart_path)
        checkpoint = {'variant': args.variant, 'architecture': 'MLP-Mixer', 'state_dict': model.state_dict(), 'test_acc': test_acc}

        torch.save(checkpoint, 'mlp_mixer_weights.pt'); print('Веса сохранены: mlp_mixer_weights.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data'); parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE); parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--warmup', type=int, default=WARMUP_EPOCHS); parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cutmix-prob', type=float, default=0.5); parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--variant', type=int, default=VARIANT); parser.add_argument('--experiment', default='CIFAR10_MLP_Mixer')
    main(parser.parse_args())