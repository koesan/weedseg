"""
DRONEQUBE Gorev 4 - WeedyRice semantic segmentation egitimi.
WeedyRice-RGBMS-DB veri setini dogrudan RGB + binary mask olarak kullanir
ve DeepLabV3+ ile semantic segmentation egitir.
"""

import argparse
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mpl-cache"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from weedyrice_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    WeedyRiceSemanticDataset,
    dice_loss,
    ensure_dir,
    extract_dataset,
    find_dataset_root,
    get_segmentation_model,
    load_metadata_index,
    load_split_lists,
    segmentation_metrics,
    summarize_binary_metrics,
    write_json,
)



def parse_args():
    parser = argparse.ArgumentParser(description="WeedyRice DeepLabV3+ egitimi")
    parser.add_argument("--dataset-zip", help="WeedyRice ZIP dosyasi")
    parser.add_argument("--dataset-dir", help="Acilmis WeedyRice klasoru")
    parser.add_argument("--save-dir", help="Cikti klasoru", default=str(BASE_DIR))
    parser.add_argument("--extract-dir", help="ZIP acma klasoru")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None, help="cuda, cpu veya bos")
    parser.add_argument("--project-name", default="weedy_rice_deeplabv3plus")
    parser.add_argument("--pretrained-backbone", action="store_true", help="Backbone icin ImageNet agirliklari kullan")
    return parser.parse_args()


def resolve_dataset(args, save_dir):
    dataset_dir = args.dataset_dir
    dataset_zip = args.dataset_zip

    if not dataset_dir and not dataset_zip:
        dataset_zip = input("Veri seti ZIP yolu: ").strip()
        save_dir_input = input(f"Cikti kayit yolu [{save_dir}]: ").strip()
        if save_dir_input:
            save_dir = Path(save_dir_input).resolve()

    if dataset_dir:
        return find_dataset_root(dataset_dir), save_dir

    if not dataset_zip:
        raise ValueError("Veri seti icin --dataset-zip veya --dataset-dir verilmelidir.")

    extract_dir = args.extract_dir or str(save_dir / "_weedy_rice_extract")
    dataset_root = extract_dataset(dataset_zip, extract_dir)
    return dataset_root, save_dir


def evaluate(model, dataloader, device, criterion_bce):
    model.eval()
    total_loss = 0.0
    batches = 0
    sums = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            loss = criterion_bce(logits, masks) + dice_loss(logits, masks)
            metrics = segmentation_metrics(logits, masks)

            total_loss += loss.item()
            batches += 1
            for key in sums:
                sums[key] += metrics[key]

    if batches == 0:
        return {"loss": 0.0, "iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}

    return {
        "loss": total_loss / batches,
        "iou": sums["iou"] / batches,
        "dice": sums["dice"] / batches,
        "precision": sums["precision"] / batches,
        "recall": sums["recall"] / batches,
    }


def denormalize_image(image_tensor):
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image * IMAGENET_STD) + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def save_epoch_preview(model, dataset, device, out_path, epoch):
    if len(dataset) == 0:
        return

    sample_idx = torch.randint(0, len(dataset), (1,)).item()
    sample = dataset[sample_idx]
    image_tensor = sample["image"].unsqueeze(0).to(device)
    gt_mask = sample["mask"][0].numpy().astype(np.uint8)

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        pred_mask = (torch.sigmoid(logits)[0, 0].detach().cpu().numpy() >= 0.5).astype(np.uint8)

    image_rgb = denormalize_image(sample["image"])

    gt_overlay = image_rgb.copy()
    gt_overlay[gt_mask == 1] = (gt_overlay[gt_mask == 1] * 0.55 + np.array([59, 130, 246]) * 0.45).astype(np.uint8)
    pred_overlay = image_rgb.copy()
    pred_overlay[pred_mask == 1] = (pred_overlay[pred_mask == 1] * 0.55 + np.array([239, 68, 68]) * 0.45).astype(np.uint8)

    diff = np.zeros_like(image_rgb)
    diff[np.logical_and(gt_mask == 1, pred_mask == 1)] = [34, 197, 94]
    diff[np.logical_and(gt_mask == 1, pred_mask == 0)] = [239, 68, 68]
    diff[np.logical_and(gt_mask == 0, pred_mask == 1)] = [245, 158, 11]

    metrics = summarize_binary_metrics(gt_mask, pred_mask)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(image_rgb); axes[0].set_title("Orijinal"); axes[0].axis("off")
    axes[1].imshow(gt_overlay); axes[1].set_title("Ground Truth"); axes[1].axis("off")
    axes[2].imshow(pred_overlay); axes[2].set_title("Tahmin"); axes[2].axis("off")
    axes[3].imshow(diff); axes[3].set_title(f"Hata\nIoU={metrics['iou']:.3f} Dice={metrics['dice']:.3f}"); axes[3].axis("off")
    plt.suptitle(f"Epoch {epoch:03d} | {sample['name']}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir).resolve()
    ensure_dir(save_dir)

    dataset_root, save_dir = resolve_dataset(args, save_dir)
    split_names = load_split_lists(dataset_root)
    metadata_index = load_metadata_index(dataset_root, split_names)

    run_dir = save_dir / "egitim" / args.project_name
    ensure_dir(run_dir)
    preview_dir = run_dir / "epoch_previews"
    ensure_dir(preview_dir)
    write_json(run_dir / "metadata_index.json", metadata_index)
    write_json(
        run_dir / "dataset_info.json",
        {
            "dataset_name": "WeedyRice-RGBMS-DB",
            "source_root": str(dataset_root),
            "imgsz": args.imgsz,
            "class_names": ["weedy_rice"],
            "splits": {key: len(value) for key, value in split_names.items()},
            "mode": "semantic-segmentation",
            "model": "deeplabv3plus",
        },
    )

    train_ds = WeedyRiceSemanticDataset(dataset_root, "train", imgsz=args.imgsz, augment=True)
    val_ds = WeedyRiceSemanticDataset(dataset_root, "val", imgsz=args.imgsz, augment=False)
    test_ds = WeedyRiceSemanticDataset(dataset_root, "test", imgsz=args.imgsz, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Veri seti: {dataset_root}")
    print(f"Cikti dizini: {run_dir}")
    print(f"Device: {device}")
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    model = get_segmentation_model("deeplabv3plus", out_channels=1, pretrained_backbone=args.pretrained_backbone).to(device)
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_iou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        step_count = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion_bce(logits, masks) + dice_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            step_count += 1

        train_loss = train_loss / max(step_count, 1)
        val_metrics = evaluate(model, val_loader, device, criterion_bce)
        scheduler.step(val_metrics["iou"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        save_epoch_preview(model, val_ds, device, preview_dir / f"epoch_{epoch:03d}.png", epoch)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_iou={val_metrics['iou']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "imgsz": args.imgsz,
                    "best_val_iou": best_iou,
                    "epoch": epoch,
                    "dataset_root": str(dataset_root),
                    "arch": "deeplabv3plus",
                },
                run_dir / "best.pt",
            )

    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, criterion_bce)

    summary = {
        "dataset": "WeedyRice-RGBMS-DB",
        "model": "deeplabv3plus",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "best_val_iou": best_iou,
        "test_loss": test_metrics["loss"],
        "test_iou": test_metrics["iou"],
        "test_dice": test_metrics["dice"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
    }

    write_json(run_dir / "history.json", history)
    write_json(run_dir / "summary.json", summary)
    shutil.copy2(run_dir / "best.pt", save_dir / "best.pt")
    shutil.copy2(run_dir / "metadata_index.json", save_dir / "metadata_index.json")
    shutil.copy2(run_dir / "dataset_info.json", save_dir / "dataset_info.json")

    print("\nTest Sonucu")
    print(summary)
    print(f"\nAgirliklar: {run_dir / 'best.pt'}")
    print(f"Kopya: {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
