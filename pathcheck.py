import os

# Paths to your dataset splits
splits = ["train", "valid", "test"]
base_dir = r"C:/Users/91984/Desktop/22881A7227/NutriLens/IndianFoodNet.v1i.yolov8"

def clean_split(split):
    img_dir = os.path.join(base_dir, split, "images")
    lbl_dir = os.path.join(base_dir, split, "labels")

    img_files = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    lbl_files = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')}

    # Find mismatches
    only_images = img_files - lbl_files
    only_labels = lbl_files - img_files

    # Delete extras
    for f in only_images:
        os.remove(os.path.join(img_dir, f + ".jpg")) if os.path.exists(os.path.join(img_dir, f + ".jpg")) else None
        os.remove(os.path.join(img_dir, f + ".jpeg")) if os.path.exists(os.path.join(img_dir, f + ".jpeg")) else None
        os.remove(os.path.join(img_dir, f + ".png")) if os.path.exists(os.path.join(img_dir, f + ".png")) else None
        print(f"🗑️ Removed unmatched image: {f}")

    for f in only_labels:
        os.remove(os.path.join(lbl_dir, f + ".txt"))
        print(f"🗑️ Removed unmatched label: {f}")

    print(f"✅ {split} cleaned: {len(img_files & lbl_files)} pairs remain")

for split in splits:
    clean_split(split)

print("\n🎯 Dataset cleaned successfully! Now every image has a label and every label has an image.")
