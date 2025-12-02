import cv2 as cv, glob, numpy as np, os

ROOT = "data/split_80_20"
CLASSES = ["abacate","aroeira","eucalipto"]
SAMPLE_ALL = True   # True -> conta todos; False -> conta uma amostra (mais rápido)
sift = cv.SIFT_create()

train_paths = []
for cls in CLASSES:
    p = sorted(glob.glob(os.path.join(ROOT, "train", cls, "*")))
    # keep only image extensions
    p = [x for x in p if x.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))]
    train_paths += p

if not train_paths:
    raise SystemExit("No training images found under data/split_80_20/train/*")

if SAMPLE_ALL:
    paths_to_check = train_paths
else:
    # sample up to 30 images uniformly
    import random
    random.seed(42)
    paths_to_check = random.sample(train_paths, min(len(train_paths), 30))

counts = []
print("Counting descriptors on", len(paths_to_check), "images...")
for p in paths_to_check:
    img = cv.imread(p, cv.IMREAD_GRAYSCALE)
    if img is None:
        print("[WARN] can't read", p); continue
    kps, des = sift.detectAndCompute(img, None)
    n = 0 if des is None else des.shape[0]
    counts.append(n)

mean_per_image = float(np.mean(counts))
median_per_image = float(np.median(counts))
total_images = len(train_paths)
estimated_total = mean_per_image * total_images

# recommendation heuristic
if estimated_total > 300_000:
    rec = 200_000
elif estimated_total > 150_000:
    rec = 100_000
elif estimated_total > 60_000:
    rec = 50_000
else:
    rec = None  # use all

print(f"Images in train: {total_images}")
print(f"Mean descriptors/image (sampled): {mean_per_image:.1f}")
print(f"Median descriptors/image (sampled): {median_per_image:.1f}")
print(f"Estimated total descriptors (mean * n_images): {estimated_total:.0f}")
print("Suggested sample_descriptors value:", rec)