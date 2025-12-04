import os
import cv2
import numpy as np

# ---------------------------------------------------------
# PATHS (Streamlit Cloud compatible)
# ---------------------------------------------------------

# Folder where user stores templates (read-only on Streamlit Cloud)
DEFAULT_TEMPLATES_DIR = "Module2/templates"

# Output directory (Streamlit Cloud only allows writing to /mount/tmp)
OUTPUT_DIR = "/mount/tmp/module2_output"

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")
THRESHOLD = 0.4
SCALE_RANGE = np.linspace(0.4, 1.8, 20)
EXPANSION_FACTOR = 1.6  # enlarge box a bit


# ---------------------------------------------------------
# Helper: Load template files
# ---------------------------------------------------------
def _load_template_files(templates_dir: str):
    """Return sorted list of valid template filenames in templates_dir."""

    print("DEBUG: TEMPLATES_DIR =", templates_dir)

    if not os.path.isdir(templates_dir):
        print("DEBUG: templates dir does NOT exist")
        return []

    files = [
        f
        for f in sorted(os.listdir(templates_dir))
        if f.lower().endswith(SUPPORTED_EXTS)
    ]

    print("DEBUG: found template files:", files)
    return files


# ---------------------------------------------------------
# Main Template Matching Function
# ---------------------------------------------------------
def match_from_database(scene_bgr, templates_dir: str | None = None):
    """
    Main function called by app.py.
    Streamlit Cloud safe (no writing outside /mount/tmp).
    """

    if scene_bgr is None:
        return None, "No scene image passed to template_matching."

    if templates_dir is None:
        templates_dir = DEFAULT_TEMPLATES_DIR

    template_files = _load_template_files(templates_dir)
    if not template_files:
        return scene_bgr, f"No templates found in folder: {templates_dir}"

    scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
    H, W = scene_gray.shape[:2]

    global_best_val = -1.0
    global_best_loc = None
    global_best_w = None
    global_best_h = None
    global_best_template = None

    # -----------------------------------------------------
    # Search over all templates & scales
    # -----------------------------------------------------
    for tmpl_name in template_files:
        tmpl_path = os.path.join(templates_dir, tmpl_name)
        tmpl_bgr = cv2.imread(tmpl_path)

        if tmpl_bgr is None:
            print("DEBUG: failed to read template:", tmpl_path)
            continue

        tmpl_gray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
        h0, w0 = tmpl_gray.shape[:2]

        for scale in SCALE_RANGE:
            new_w = int(w0 * scale)
            new_h = int(h0 * scale)
            if new_w < 5 or new_h < 5:
                continue

            resized = cv2.resize(
                tmpl_gray, (new_w, new_h), interpolation=cv2.INTER_AREA
            )

            h, w = resized.shape[:2]
            if h > H or w > W:
                continue

            res = cv2.matchTemplate(
                scene_gray, resized, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > global_best_val:
                global_best_val = max_val
                global_best_loc = max_loc
                global_best_w = w
                global_best_h = h
                global_best_template = tmpl_name

    result_img = scene_bgr.copy()

    # No good match found
    if (
        global_best_val is None
        or global_best_loc is None
        or global_best_w is None
        or global_best_h is None
        or global_best_val < THRESHOLD
    ):
        msg = f"No match found. Best score = {global_best_val:.2f}"
        return result_img, msg

    # -----------------------------------------------------
    # Draw bounding box + label
    # -----------------------------------------------------
    x, y = global_best_loc
    w, h = global_best_w, global_best_h

    cx = x + w // 2
    cy = y + h // 2

    new_w = int(w * EXPANSION_FACTOR)
    new_h = int(h * EXPANSION_FACTOR)

    x1 = max(cx - new_w // 2, 0)
    y1 = max(cy - new_h // 2, 0)
    x2 = min(cx + new_w // 2, W)
    y2 = min(cy + new_h // 2, H)

    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    label = f"{global_best_template} ({global_best_val:.2f})"
    cv2.putText(
        result_img,
        label,
        (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    # -----------------------------------------------------
    # Save output image (ONLY in /mount/tmp)
    # -----------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_name = f"result_{os.path.splitext(global_best_template)[0]}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    cv2.imwrite(out_path, result_img)
    print("DEBUG: saved result to:", out_path)

    msg = f"Found: {global_best_template} (score={global_best_val:.2f})"
    return result_img, msg
