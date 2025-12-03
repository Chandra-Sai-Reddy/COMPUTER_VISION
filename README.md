# CSC 8830 – Computer Vision Portfolio

This repository contains my assignments for CSC 8830 (Computer Vision), implemented in Python using Streamlit.

## Structure

- `app.py` – Main Streamlit dashboard for all modules.
- `Module1/` – Single View Metrology
- `Module2/` – Object Detection (Templates)
- `Module3/` – Features & Segmentation
- `Module4/` – Stitching & SIFT
- `Module5_6/` – Motion Tracking
- `Module7/` – Stereo & Pose Estimation

## How to run locally

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit app
streamlit run app.py

