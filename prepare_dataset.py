from pathlib import Path

dataset = Path(r"E:\Image search\dataset")
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

if not dataset.exists():
    raise FileNotFoundError(f"Dataset not found: {dataset}")

for folder in sorted(dataset.iterdir()):
    if not folder.is_dir():
        continue

    images = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    for image in images[50:]:
        image.unlink()

    print(
        f"{folder.name}: kept {min(50, len(images))}, "
        f"deleted {max(0, len(images) - 50)}"
    )

print("Done! Dataset reduced.")