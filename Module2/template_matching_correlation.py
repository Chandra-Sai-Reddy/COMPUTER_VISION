import os
import cv2
import numpy as np

# === Correct Paths ===
BASE_DIR = "/Users/donthireddychandrasaireddy/Desktop/COMPUTER_VISION/Module2/template_matching_project"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Parameters ===
THRESHOLD = 0.4  # Lower threshold for cross-scene templates
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")
SCALE_RANGE = np.linspace(0.3, 2.0, 20)  # Wider range for better matching
EXPANSION_FACTOR = 1.6  # Enlarges bounding box around matched region

# === Load image and template file names ===
image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(SUPPORTED_EXTS)])
template_files = sorted([f for f in os.listdir(TEMPLATES_DIR) if f.lower().endswith(SUPPORTED_EXTS)])

# === Ensure equal count ===
if len(image_files) != len(template_files):
    print("⚠️ The number of images and templates are not equal! Check your folders.")
else:
    print(f"Found {len(image_files)} image-template pairs.\n")

# === Perform 1-to-1 multi-scale template matching ===
for i, (img_name, template_name) in enumerate(zip(image_files, template_files), 1):
    img_path = os.path.join(IMAGES_DIR, img_name)
    template_path = os.path.join(TEMPLATES_DIR, template_name)

    # Load image and template
    image = cv2.imread(img_path)
    template = cv2.imread(template_path)
    if image is None or template is None:
        print(f"⚠️ Skipping pair {i}: Cannot read image or template.")
        continue

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    orig_h, orig_w = template_gray.shape[:2]

    best_val = -1
    best_loc = None
    best_w, best_h = None, None

    # === Multi-scale template matching ===
    for scale in SCALE_RANGE:
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        if new_w < 5 or new_h < 5:  # Skip too small templates
            continue
        resized_template = cv2.resize(template_gray, (new_w, new_h))
        res = cv2.matchTemplate(gray_img, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_w, best_h = new_w, new_h

    print(f"[{i}] Image: {img_name} | Template: {template_name} | Best correlation: {best_val:.3f}")

    # === Draw bounding box if best match exceeds threshold ===
    if best_val >= THRESHOLD:
        top_left = best_loc
        bottom_right = (top_left[0] + best_w, top_left[1] + best_h)

        # --- Expand bounding box for better visualization ---
        center_x = top_left[0] + best_w // 2
        center_y = top_left[1] + best_h // 2
        new_w = int(best_w * EXPANSION_FACTOR)
        new_h = int(best_h * EXPANSION_FACTOR)
        top_left = (max(center_x - new_w // 2, 0), max(center_y - new_h // 2, 0))
        bottom_right = (min(center_x + new_w // 2, image.shape[1]), 
                        min(center_y + new_h // 2, image.shape[0]))

        # Draw bounding box and label
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
        label = f"{template_name} ({best_val:.2f})"
        cv2.putText(image, label, (top_left[0], max(top_left[1]-10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        label = f"No match ({best_val:.2f})"
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # === Save result ===
    output_name = f"result_{os.path.splitext(img_name)[0]}_{os.path.splitext(template_name)[0]}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    cv2.imwrite(output_path, image)

print("\n✅ Multi-scale template matching completed!")
print(f"📁 Check results in: {OUTPUT_DIR}")
