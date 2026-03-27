import os

base = r"C:\Users\91984\Desktop\22881A7227\NutriLens\IndianFoodNet.v1.yolov8"
splits = ["train", "valid", "test"]

for split in splits:
    img_dir = os.path.join(base, split, "images")
    lbl_dir = os.path.join(base, split, "labels")

    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        continue

    img_files = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))}
    lbl_files = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')}

    only_images = img_files - lbl_files
    only_labels = lbl_files - img_files

    print(f"\nChecking {split}:")
    print(" Images without labels:", len(only_images))
    print(" Labels without images:", len(only_labels))

    # Delete extra images
    for f in only_images:
        for ext in ['.jpg', '.png', '.jpeg']:
            path = os.path.join(img_dir, f + ext)
            if os.path.exists(path):
                os.remove(path)

    # Delete extra labels
    for f in only_labels:
        path = os.path.join(lbl_dir, f + ".txt")
        if os.path.exists(path):
            os.remove(path)

print("\n✅ Dataset cleaned successfully. All images now have matching labels.")
