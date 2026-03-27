#!/usr/bin/env python3
"""
prepare_data_use_valid.py
Auto-generate data.yaml for YOLOv8 using exact split names:
 - expects dataset root to have 'train', 'test' and/or 'valid' (preferred)
 - will use 'valid' if present and set data.yaml 'val' -> 'valid/images'
 - supports images/labels inside split (either subfolders or directly inside)
 - extracts class ids from YOLO .txt labels, removes unused classes, maps names if classes file exists
"""

import os, shutil, yaml
from pathlib import Path

# ====== Update this to your dataset folder (already set for you) ======
dataset_root = r"C:\Users\91984\Desktop\22881A7227\NutriLens\IndianFoodNet.v1i.yolov8"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def is_image_file(name):
    return name.lower().endswith(IMG_EXTS)

def find_split_exact(root, name):
    p = os.path.join(root, name)
    return p if os.path.isdir(p) else None

def locate_images_labels(split_path):
    """Return (images_dir, labels_dir) for a split.
       Prefer subfolders named 'images'/'images*' and 'labels'/'label*', else use split_path if it directly contains files.
    """
    if split_path is None:
        return None, None
    images_dir = None
    labels_dir = None
    for entry in os.listdir(split_path):
        p = os.path.join(split_path, entry)
        if os.path.isdir(p):
            ln = entry.lower()
            if "image" in ln and images_dir is None:
                images_dir = p
            if "label" in ln and labels_dir is None:
                labels_dir = p
    # fallback: if no images subfolder, maybe images are directly in split_path
    if images_dir is None:
        if any(is_image_file(f) for f in os.listdir(split_path)):
            images_dir = split_path
    # fallback: if no labels subfolder, maybe labels are directly in split_path
    if labels_dir is None:
        if any(f.lower().endswith(".txt") for f in os.listdir(split_path)):
            labels_dir = split_path
    return images_dir, labels_dir

def ensure_dir(path):
    if path is None:
        return
    os.makedirs(path, exist_ok=True)

def collect_class_ids(labels_dir):
    ids = set()
    if labels_dir is None:
        return ids
    for root, _, files in os.walk(labels_dir):
        for fname in files:
            if not fname.lower().endswith(".txt"): 
                continue
            fp = os.path.join(root, fname)
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if not parts: 
                            continue
                        try:
                            ids.add(int(parts[0]))
                        except:
                            print(f"⚠ Non-integer class id in {fp}: {parts[0]}")
            except Exception as e:
                print(f"⚠ Could not read {fp}: {e}")
    return ids

def find_class_names_file(search_roots):
    candidates = ["classes.txt", "obj.names", "names.txt", "classes.names"]
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for cand in candidates:
            fp = os.path.join(root, cand)
            if os.path.exists(fp):
                try:
                    with open(fp, "r", encoding="utf-8") as fh:
                        lines = [l.strip() for l in fh.readlines() if l.strip()]
                        if lines:
                            return lines
                except:
                    pass
    return None

def count_labels(labels_dir, final_ids):
    counts = {cid:0 for cid in final_ids}
    if not labels_dir or not os.path.exists(labels_dir):
        return counts
    for root, _, files in os.walk(labels_dir):
        for fname in files:
            if not fname.lower().endswith(".txt"): 
                continue
            fp = os.path.join(root, fname)
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            cid = int(parts[0])
                            if cid in counts:
                                counts[cid] += 1
                        except:
                            pass
            except:
                pass
    return counts

def ensure_val_example(val_images_dir, val_labels_dir, train_images_dir, train_labels_dir, final_ids):
    # If val/images empty, try copy one image+label from train or create a dummy
    if not os.path.exists(val_images_dir):
        os.makedirs(val_images_dir, exist_ok=True)
    if not os.path.exists(val_labels_dir):
        os.makedirs(val_labels_dir, exist_ok=True)
    imgs = [f for f in os.listdir(val_images_dir) if is_image_file(f)]
    if imgs:
        return
    # try copy
    if train_images_dir and os.path.exists(train_images_dir):
        timgs = [f for f in os.listdir(train_images_dir) if is_image_file(f)]
        if timgs:
            src = os.path.join(train_images_dir, timgs[0])
            dst = os.path.join(val_images_dir, timgs[0])
            shutil.copy2(src, dst)
            lblname = os.path.splitext(timgs[0])[0] + ".txt"
            src_lbl = None
            if train_labels_dir and os.path.exists(train_labels_dir):
                cand = os.path.join(train_labels_dir, lblname)
                if os.path.exists(cand):
                    src_lbl = cand
            if src_lbl:
                shutil.copy2(src_lbl, os.path.join(val_labels_dir, lblname))
            else:
                if final_ids:
                    with open(os.path.join(val_labels_dir, lblname), "w", encoding="utf-8") as fh:
                        fh.write(f"{final_ids[0]} 0.5 0.5 0.2 0.2\n")
            print(f"✅ Copied one train example to valid: {dst}")
            return
    # fallback create dummy (Pillow optional)
    try:
        from PIL import Image
        dummy = Image.new("RGB",(640,640),(0,0,0))
        dummy_fp = os.path.join(val_images_dir, "dummy.jpg")
        dummy.save(dummy_fp, "JPEG")
        lbl_fp = os.path.join(val_labels_dir, "dummy.txt")
        if final_ids:
            with open(lbl_fp, "w", encoding="utf-8") as fh:
                fh.write(f"{final_ids[0]} 0.5 0.5 0.2 0.2\n")
        else:
            open(lbl_fp, "a").close()
        print("✅ Created dummy image+label in valid to prevent YOLO errors.")
    except Exception:
        open(os.path.join(val_images_dir, "dummy.jpg"), "a").close()
        open(os.path.join(val_labels_dir, "dummy.txt"), "a").close()
        print("⚠ Pillow not available — created placeholder files in valid.")

def main():
    print(f"Dataset root = {dataset_root}")
    if not os.path.exists(dataset_root):
        raise SystemExit("ERROR: dataset_root does not exist.")

    # Prefer exact folders: 'train', 'valid', 'test' (user said valid exists)
    train_split = find_split_exact(dataset_root, "train") or find_split_exact(dataset_root, "Train")
    valid_split = find_split_exact(dataset_root, "valid") or find_split_exact(dataset_root, "valid") or find_split_exact(dataset_root, "Valid")
    # also accept "validation" fallback
    if valid_split is None:
        valid_split = find_split_exact(dataset_root, "validation") or find_split_exact(dataset_root, "Validation")
    test_split = find_split_exact(dataset_root, "test") or find_split_exact(dataset_root, "Test")

    # fallback: try to detect any candidate train if not found
    if train_split is None:
        for name in os.listdir(dataset_root):
            cand = os.path.join(dataset_root, name)
            if os.path.isdir(cand):
                imgs,lbls = locate_images_labels(cand)
                if imgs and lbls:
                    train_split = cand
                    print(f"⚠ Using '{cand}' as train (fallback).")
                    break
    if train_split is None:
        raise SystemExit("ERROR: Could not locate a train split under dataset root.")

    print("Using train split:", train_split)
    print("Detected valid split:", valid_split)
    print("Detected test split:", test_split)

    train_images, train_labels = locate_images_labels(train_split)
    val_images, val_labels = (None, None)
    if valid_split:
        val_images, val_labels = locate_images_labels(valid_split)
    else:
        # create valid folder inside dataset_root if not present
        valid_split = os.path.join(dataset_root, "valid")
        val_images = os.path.join(valid_split, "images")
        val_labels = os.path.join(valid_split, "labels")

    # if test exists, locate images/labels
    test_images, test_labels = (None, None)
    if test_split:
        test_images, test_labels = locate_images_labels(test_split)

    # ensure directories exist (create valid if needed)
    ensure_dir(train_images); ensure_dir(train_labels)
    ensure_dir(valid_split); ensure_dir(val_images); ensure_dir(val_labels)
    ensure_dir(test_images); ensure_dir(test_labels)

    print("train images:", train_images)
    print("train labels:", train_labels)
    print("valid images:", val_images)
    print("valid labels:", val_labels)
    if test_images:
        print("test images:", test_images)
        print("test labels:", test_labels)

    # collect class ids
    train_ids = collect_class_ids(train_labels)
    val_ids = collect_class_ids(val_labels)
    all_ids = sorted(train_ids.union(val_ids))
    print("Detected class ids:", all_ids)

    # find mapping file if any
    classes_file_lines = find_class_names_file([dataset_root, train_split, train_labels, val_labels, test_split, test_labels])
    if classes_file_lines:
        print("Loaded class names file with", len(classes_file_lines), "names.")

    used_ids = sorted(list(train_ids.union(val_ids)))
    if not used_ids:
        print("⚠ No class ids found in labels. Stopping.")
        # still produce a minimal data.yaml to avoid errors
    else:
        print("Used class ids:", used_ids)

    # map ids->names
    final_ids = used_ids
    if classes_file_lines and len(classes_file_lines) >= max(final_ids)+1 if final_ids else False:
        class_names = [classes_file_lines[i] for i in final_ids]
    else:
        # if train_images has subfolders matching number of ids, use those
        class_names = None
        if train_images and os.path.isdir(train_images):
            subs = sorted([d for d in os.listdir(train_images) if os.path.isdir(os.path.join(train_images,d))])
            if subs and len(subs) == len(all_ids):
                try:
                    class_names = [subs[i] for i in final_ids]
                except:
                    class_names = None
        if class_names is None:
            class_names = [f"class{cid}" for cid in final_ids]

    # ensure valid has example so YOLO doesn't error
    ensure_val_example(val_images, val_labels, train_images, train_labels, final_ids)

    # counts
    train_counts = count_labels(train_labels, final_ids)
    val_counts = count_labels(val_labels, final_ids)

    # write data.yaml with val pointing to valid/images (relative)
    def rel(p):
        if p is None: return ""
        try:
            return os.path.relpath(p, dataset_root).replace("\\","/")
        except:
            return p.replace("\\","/")

    train_rel = rel(train_images) or "train/images"
    valid_rel = rel(val_images) or "valid/images"
    test_rel = rel(test_images) or valid_rel

    data = {
        "path": dataset_root.replace("\\","/"),
        "train": train_rel,
        "val": valid_rel,      # points to 'valid/images' relative path if valid exists
        "test": test_rel,
        "nc": len(class_names),
        "names": class_names
    }

    yaml_path = os.path.join(dataset_root, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print("\n✅ data.yaml written to:", yaml_path)
    print("  nc:", data["nc"])
    print("  train:", data["train"])
    print("  val:", data["val"])
    print("  test:", data["test"])
    print("  names:", class_names)
    print("\n📊 label counts (train / valid):")
    for cid, cname in zip(final_ids, class_names):
        print(f"  {cid} - {cname}: train={train_counts.get(cid,0)}, valid={val_counts.get(cid,0)}")

if __name__ == "__main__":
    main()
