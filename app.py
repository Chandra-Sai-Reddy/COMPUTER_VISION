import streamlit as st
import cv2
import numpy as np
import pandas as pd 
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import sys
# import mediapipe as mp

# --- IMPORT CHECK ---
# Make sure your folders inside 'modules' are actually named 'Module1', 'Module2', etc.
import sys
sys.path.append(".")

from Module1 import measure_tool
from Module2 import template_matching, fourier
from Module3 import features
from Module4.task1 import panorama
from Module4.task2 import sift_scratch
from Module5_6 import tracker
from Module7 import stereo_pose

# PAGE CONFIGURATION
st.set_page_config(page_title="CV Dashboard", layout="wide")

# SIDEBAR NAVIGATION
st.sidebar.title("CV Modules Dashboard")
module_selection = st.sidebar.radio(
    "Go to:",
    [
        "Home",
        "1. Single View Metrology",
        "2. Object Detection (Templates)",
        "3. Features & Segmentation",
        "4. Stitching & SIFT",
        "5. Motion Tracking",
        "6. Stereo & Pose Estimation"
    ]
)
# --- HOME PAGE ---
if module_selection == "Home":
    st.title("Computer Vision Portfolio")
    st.write("Welcome to the unified dashboard for CSC 8830 assignments.")
    st.info("Select a module from the sidebar to begin.")

    st.subheader("📚 Modules")

    st.markdown(
        """
**Module 1 – Single View Metrology**    
🎥 [Watch Module 1 Video](https://drive.google.com/file/d/1SQbD7FMgIOBTwzA18kj3R8ezSXbvoX06/view?usp=sharing)

**Module 2 – Object Detection (Templates)**    
🎥 [Watch Module 2 Video](https://drive.google.com/file/d/1EoQq1CFXyQ3gdXLaQwqUEgwul2PoLQsf/view?usp=sharing)

**Module 3 – Features & Segmentation**    
🎥 [Watch Module 3 Video](https://drive.google.com/file/d/16-lQ2WF_qrPpoSuhrr6J99QkGICZA-sR/view?usp=sharing)

**Module 4 – Stitching & SIFT**   
🎥 [Watch Module 4a Video](https://drive.google.com/file/d/1au9CoUl4wbV9XiUYYbqKQt104xBfEtQg/view?usp=sharing)
🎥 [Watch Module 4b Video](https://drive.google.com/file/d/1oEI53ZfvLeQ3XmBOZYjFjLua9a6W1_Qz/view?usp=sharing)

**Module 5 & 6 – Motion Tracking**  
🎥 [Watch Module 5 & 6a Video](https://drive.google.com/file/d/1RzW41atJhIvvctleRWjup967KLK9_6Hq/view?usp=sharing)
🎥 [Watch Module 5 & 6b Video](https://drive.google.com/file/d/1Tsv0kS4tJk8666dqMjvglX3Kupw37TiZ/view?usp=sharing)

**Module 7 – Stereo & Pose Estimation**  
🎥 [Watch Module 7 Video](https://drive.google.com/file/d/105THyo8xc1iBKanJlfoAsCd-8Jn6_5Ze/view?usp=sharing)


**Module 8 – Table **  
[Module 8 Table:  ](https://docs.google.com/document/d/1UD0CUdxO_GDj260lJ8xRvxfevBcHpFfhlbp6cVLhzYE/edit?usp=sharing)
        """
    )

# ─────────────────────────────────────────────
# MODULE 1: SINGLE VIEW METROLOGY
# ─────────────────────────────────────────────
if module_selection == "1. Single View Metrology":
    st.header("1. Single View Metrology")

    # 1. Load Calibration
    # Ensure calibration_data_mac.npz exists in the root (same folder as app.py),
    # or adjust the path here.
    try:
        mtx, dist, f = measure_tool.get_calibration_data("calibration_data_mac.npz")
    except Exception:
        f = None

    if f is None:
        st.error("⚠️ 'calibration_data_mac.npz' not found! Using default f = 700.0")
        f = 0.0
        is_calibrated = False
    else:
        st.success(f"Calibration Loaded. Focal Length (f) = {f:.2f}")
        is_calibrated = True

    # 2. Inputs
    col1, col2 = st.columns(2)
    with col1:
        Z = st.number_input(
            "Distance from Camera (Z) in cm:",
            min_value=1.0,
            value=50.0
        )
    with col2:
        st.info("Draw a line on the image to measure real dimensions.")

    # 3. Image Source Selection
    input_source = st.radio(
        "Select Image Source:",
        ("Upload Image", "Take Photo"),
        horizontal=True
    )

    img_file = None
    if input_source == "Upload Image":
        img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    else:
        img_file = st.camera_input("Take a picture of the object")

    # 4. Process Image & Canvas
    if img_file is not None:
        # Convert upload/camera buffer to OpenCV image
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        if img is None:
            st.error("Could not decode the image. Please try another file.")
        else:
            # Optional: undistort using calibration
            if is_calibrated:
                img = measure_tool.undistort_image(img, mtx, dist)
                st.caption("Image Undistorted using Calibration Data")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            # 4a. Interactive Canvas OVER the image
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",   # semi-transparent orange
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=Image.fromarray(img_rgb),  # show image in canvas
                update_streamlit=True,
                height=int(h),
                width=int(w),
                drawing_mode="line",
                key="canvas_m1",
            )

            # 5. Read last drawn object and compute real length
            if canvas_result.json_data is not None:
                objects = pd.json_normalize(canvas_result.json_data.get("objects", []))

                if not objects.empty:
                    # Take the last drawn object
                    obj = objects.iloc[-1]
                    x1, y1 = obj["left"], obj["top"]
                    x2 = x1 + obj["width"] * obj["scaleX"]
                    y2 = y1 + obj["height"] * obj["scaleY"]

                    real_size = measure_tool.calculate_real_distance(
                        (x1, y1),
                        (x2, y2),
                        Z,
                        f
                    )
                    st.metric("Calculated Real Size", f"{real_size:.2f} cm")
                else:
                    st.warning("Draw a line on the image to measure its real-world length.")
# --- MODULE 2: OBJECT DETECTION & FOURIER ---
elif module_selection == "2. Object Detection (Templates)":
    st.header("2. Object Detection & Fourier Transform")
    
    tab1, tab2 = st.tabs(["Task 1: Template Matching (Database)", "Task 2: Fourier Restoration"])
    
    # --- TASK 1: TEMPLATE MATCHING (DATABASE SCAN) ---
    with tab1:
        st.subheader("Detect Object from Local Database")
        st.info("Upload a Scene. The app will auto-scan the templates folder to find the object.")
        
        # DEBUG: show which template_matching file & default folder are used
        st.text(f"template_matching module: {template_matching.__file__}")
        st.text(f"Default templates dir: {template_matching.DEFAULT_TEMPLATES_DIR}")

        # If you want, you can also allow overriding the path via UI:
        templates_dir_ui = st.text_input(
            "Templates folder on disk:",
            value=template_matching.DEFAULT_TEMPLATES_DIR
        )
        
        scene_file = st.file_uploader("Upload Scene Image", type=['jpg', 'png', 'jpeg'], key="scene_db")
        
        if st.button("Scan Scene for Objects"):
            if scene_file:
                # Decode image
                s_bytes = np.asarray(bytearray(scene_file.read()), dtype=np.uint8)
                scene = cv2.imdecode(s_bytes, 1)
                
                st.write("Scanning database... (Multi-scale matching)")
                
                # Call your template matching logic, with explicit path
                result_img, msg = template_matching.match_from_database(
                    scene,
                    templates_dir=templates_dir_ui
                )
                
                if "Found" in msg:
                    st.success(msg)
                    st.image(result_img, channels="BGR", caption="Result: Object Detected & Pixelated")
                else:
                    st.warning(msg)
                    st.image(scene, channels="BGR", caption="Original (No Match Found)")
            else:
                st.warning("Please upload a Scene image.")

    # --- TASK 2: FOURIER RESTORATION ---
    with tab2:
        st.subheader("Fourier Image Restoration")
        st.write("Apply Gaussian Blur, then recover the original using Wiener Deconvolution.")
        
        fourier_file = st.file_uploader("Upload Image to Blur & Restore", type=['jpg', 'png', 'jpeg'], key="fft")
        
        if st.button("Run Fourier Analysis"):
            if fourier_file:
                f_bytes = np.asarray(bytearray(fourier_file.read()), dtype=np.uint8)
                original = cv2.imdecode(f_bytes, 1)
                
                blurred, restored = fourier.apply_blur_and_restore(original)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(original, channels="BGR", caption="1. Original")
                with c2:
                    st.image(blurred, channels="BGR", caption="2. Heavy Blur (Input)")
                with c3:
                    st.image(restored, channels="BGR", caption="3. Restored (Output)")
            else:
                st.warning("Please upload an image.")

# --- MODULE 3: FEATURES & SEGMENTATION ---
elif module_selection == "3. Features & Segmentation":
    st.header("3. Features & Segmentation")
    
    task_mode = st.radio(
    "Select Task:",
    [
        "Task 1: Gradient & LoG", 
        "Task 2: Keypoints (Edges/Corners)", 
        "Task 3: Object Boundary",
        "Task 4: ArUco Boundary (Non-Rectangular)",
        "Task 5: SAM2 Segmentation (Compare)"
    ],
    horizontal=True
)

    
    st.divider()
    input_source = st.radio("Select Image Source:", ("Upload Image", "Take Photo"), horizontal=True)
    
    img_file = None
    if input_source == "Upload Image":
        img_file = st.file_uploader("Upload Image for Analysis", type=['jpg', 'png', 'jpeg'])
    else:
        img_file = st.camera_input("Take a picture for Analysis")
    
    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=400)
        st.divider() 
        
        if task_mode == "Task 1: Gradient & LoG":
            st.subheader("Gradient Analysis")
            mag, angle, log = features.get_gradient_and_log(img)
            col1, col2, col3 = st.columns(3)
            with col1: st.image(mag, caption="Gradient Magnitude")
            with col2: st.image(angle, caption="Gradient Angle")
            with col3: st.image(log, caption="Laplacian of Gaussian")

        elif task_mode == "Task 2: Keypoints (Edges/Corners)":
            st.subheader("Keypoint Detection")
            canny, corners = features.get_keypoints(img)
            col1, col2 = st.columns(2)
            with col1: st.image(canny, caption="Canny Edges")
            with col2: st.image(cv2.cvtColor(corners, cv2.COLOR_BGR2RGB), caption="Harris Corners")

        elif task_mode == "Task 3: Object Boundary":
            st.subheader("Boundary Detection")
            mask, boundary = features.get_boundary(img)
            col1, col2 = st.columns(2)
            with col1: st.image(mask, caption="Threshold Mask")
            with col2: st.image(cv2.cvtColor(boundary, cv2.COLOR_BGR2RGB), caption="Object Boundary")
        elif task_mode == "Task 4: ArUco Boundary (Non-Rectangular)":
            st.subheader("Task 4: ArUco-Based Boundary")
            st.info("Image must contain ArUco markers on the object boundary.")

            hull_img, msg = features.get_aruco_hull(img)
            if hull_img is None:
                st.error(msg)
            else:
                st.success(msg)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
                with col2:
                    st.image(cv2.cvtColor(hull_img, cv2.COLOR_BGR2RGB),
                             caption="ArUco Convex Hull", use_column_width=True)

        elif task_mode == "Task 5: SAM2 Segmentation (Compare)":
            st.subheader("Task 5: SAM2 Segmentation (Comparison)")
            st.info("Uses ArUco marker centers as prompts for the SAM2 model.")

            with st.spinner("Running ArUco + SAM2 (this may take a moment the first time)..."):
                mask_img, overlay_img, msg = features.get_sam2_segmentation(img)

            if mask_img is None or overlay_img is None:
                st.error(msg)
            else:
                st.success(msg)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                             caption="Original", use_column_width=True)
                with col2:
                    st.image(mask_img, caption="SAM2 Mask", use_column_width=True)
                with col3:
                    st.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB),
                             caption="Overlay (Mask + Prompts)", use_column_width=True)


# --- MODULE 4: STITCHING & SIFT ---
elif module_selection == "4. Stitching & SIFT":
    st.header("4. Image Stitching & SIFT")
    
    tab1, tab2 = st.tabs(["Task 1: Panorama Stitching", "Task 2: SIFT & RANSAC Comparison"])
    
    # --- TAB 1: STITCHING ---
    with tab1:
        st.subheader("Panorama Generation")
        st.write("Requirements: Stitch 4+ images and compare with mobile panorama.")
        
        col_up1, col_up2 = st.columns(2)
        
        with col_up1:
            st.markdown("#### 1. Upload Input Images")
            uploaded_files = st.file_uploader(
                "Upload overlapping images (Bottom to Top)", 
                accept_multiple_files=True, 
                type=['jpg', 'png', 'jpeg'], 
                key="stitch_inputs"
            )
        
        with col_up2:
            st.markdown("#### 2. Upload Mobile Panorama")
            ground_truth_file = st.file_uploader(
                "Upload the panorama taken by your phone", 
                type=['jpg', 'png', 'jpeg'], 
                key="stitch_gt"
            )

        if st.button("Run Stitching"):
            if uploaded_files and len(uploaded_files) >= 2:
                st.info(f"Processing {len(uploaded_files)} images...")
                
                images_list = []
                for uploaded_file in uploaded_files:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, 1)
                    images_list.append(img)
                
                result, message = panorama.stitch_collection(images_list)
                
                if result is not None:
                    st.success("Stitching Successful!")
                    
                    st.subheader("Comparison: My Algorithm vs. Mobile Camera")
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.image(result, channels="BGR", caption="Result: My Stitching Algorithm")
                    
                    with comp_col2:
                        if ground_truth_file:
                            gt_bytes = np.asarray(bytearray(ground_truth_file.read()), dtype=np.uint8)
                            gt_img = cv2.imdecode(gt_bytes, 1)
                            st.image(gt_img, channels="BGR", caption="Ground Truth: Mobile Panorama")
                        else:
                            st.warning("Upload a Mobile Panorama to see the comparison.")
                else:
                    st.error(f"Error: {message}")
            else:
                st.warning("Please upload at least 2 images to stitch.")

    # --- TAB 2: SIFT & RANSAC ---
    with tab2:
        st.subheader("SIFT Feature Extraction & RANSAC")
        st.write("Compare 'From Scratch' implementation vs. OpenCV.")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            file1 = st.file_uploader("Image 1 (Source)", type=['jpg', 'png'], key="sift1")
        with col_s2:
            file2 = st.file_uploader("Image 2 (Destination)", type=['jpg', 'png'], key="sift2")

        if st.button("Run SIFT Comparison") and file1 and file2:
            b1 = np.asarray(bytearray(file1.read()), dtype=np.uint8)
            img1 = cv2.imdecode(b1, 1)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            
            b2 = np.asarray(bytearray(file2.read()), dtype=np.uint8)
            img2 = cv2.imdecode(b2, 1)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            st.write("Processing... This may take a moment.")
            
            c_custom, c_opencv = st.columns(2)
            
            # --- LEFT: YOUR IMPLEMENTATION ---
            with c_custom:
                st.markdown("### My Implementation")
                kp1, des1 = sift_scratch.run_from_scratch_sift(gray1)
                kp2, des2 = sift_scratch.run_from_scratch_sift(gray2)
                
                matches = sift_scratch.match_features(des1, des2)
                if len(matches) > 4:
                    H, inliers = sift_scratch.from_scratch_ransac(kp1, kp2, matches)
                    vis_custom = sift_scratch.draw_matches_custom(img1, kp1, img2, kp2, matches, inliers)
                    st.image(vis_custom, channels="BGR", caption=f"My RANSAC ({len(inliers)} Inliers)")
                else:
                    st.warning("Not enough matches found in custom implementation.")

            # --- RIGHT: OPENCV IMPLEMENTATION ---
            with c_opencv:
                st.markdown("### OpenCV Implementation")
                sift_cv = cv2.SIFT_create()
                kp1_cv, des1_cv = sift_cv.detectAndCompute(gray1, None)
                kp2_cv, des2_cv = sift_cv.detectAndCompute(gray2, None)
                
                bf = cv2.BFMatcher()
                matches_cv = bf.knnMatch(des1_cv, des2_cv, k=2)
                
                good_cv = []
                for m, n in matches_cv:
                    if m.distance < 0.75 * n.distance:
                        good_cv.append(m)
                
                if len(good_cv) > 4:
                    src_pts = np.float32([kp1_cv[m.queryIdx].pt for m in good_cv]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2_cv[m.trainIdx].pt for m in good_cv]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    matchesMask = mask.ravel().tolist()
                    
                    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
                    vis_cv = cv2.drawMatches(img1, kp1_cv, img2, kp2_cv, good_cv, None, **draw_params)
                    st.image(vis_cv, channels="BGR", caption=f"OpenCV RANSAC ({sum(matchesMask)} Inliers)")
                else:
                    st.warning("Not enough matches found in OpenCV.")
# --- MODULE 5: MOTION TRACKING ---
elif module_selection == "5. Motion Tracking":
    st.header("5. Motion Tracking")
    
    track_mode = st.radio("Select Tracking Method:", 
                          ["1. Marker-based (ArUco)", 
                           "2. Marker-less (CSRT)", 
                           "3. SAM2 Segmentation Demo"])

    # --- MODE 1: MARKER BASED ---
    if track_mode == "1. Marker-based (ArUco)":
        st.subheader("Marker-based Tracking")
        st.info("Show an ArUco 4x4 Marker to the camera.")
        
        if st.checkbox("Start Webcam (Marker)"):
            cap = cv2.VideoCapture(0)
            st_frame = st.image([])
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                output = tracker.track_markers_aruco(frame)
                st_frame.image(output, channels="BGR")
            cap.release()

    # --- MODE 2: MARKER-LESS (CSRT) ---
    elif track_mode == "2. Marker-less (CSRT)":
        st.subheader("Marker-less Object Tracking")
        st.info("1. Take photo. 2. Draw box. 3. START. (Keep object still between photo and start!)")

        # Initialize Session State
        if 'track_init_frame' not in st.session_state: st.session_state['track_init_frame'] = None
        if 'track_bbox' not in st.session_state: st.session_state['track_bbox'] = None

        col_controls, col_display = st.columns([1, 2])
        
        with col_controls:
            # 1. Take Photo
            img_buffer = st.camera_input("Take a photo to define object")
            
            if img_buffer is not None:
                bytes_data = img_buffer.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                # CRITICAL FIX 1: Resize to standard webcam resolution (640x480)
                # This ensures coordinates from photo match the video stream later
                cv2_img = cv2.resize(cv2_img, (640, 480))
                
                # CRITICAL FIX 2: Convert to RGB for consistency
                st.session_state['track_init_frame'] = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        # 2. Draw Box
        if st.session_state['track_init_frame'] is not None:
            st.markdown("### Draw a box around the object:")
            
            # Display Canvas
            # We enforce fixed 640x480 dimensions to match the resize above
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=Image.fromarray(st.session_state['track_init_frame']),
                height=480,
                width=640,
                drawing_mode="rect",
                key="tracker_canvas"
            )

            # Check for drawing
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if len(objects) > 0:
                    obj = objects[-1]
                    # No scaling needed because canvas = image size = 640x480
                    x = int(obj["left"])
                    y = int(obj["top"])
                    w = int(obj["width"])
                    h = int(obj["height"])
                    
                    if w > 10 and h > 10:
                        st.session_state['track_bbox'] = (x, y, w, h)
            
            # Show Status
            if st.session_state['track_bbox']:
                st.success(f"Target Locked: {st.session_state['track_bbox']}")
                
                # 3. Start Tracking
                if st.button("Start Tracking"):
                    st.warning("Starting... Don't move the object yet!")
                    
                    cap = cv2.VideoCapture(0)
                    
                    # Force Webcam to 640x480 to match our photo
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    # Warmup
                    for _ in range(10): cap.read()
                    
                    ret, frame = cap.read()
                    if ret:
                        # CRITICAL FIX 3: Ensure Frame is RGB to match the Photo
                        # OpenCV gives BGR, we convert to RGB before initializing tracker
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        bbox = st.session_state['track_bbox']
                        tracker_obj = tracker.MarkerlessTracker(frame_rgb, bbox)
                        
                        st_track_window = st.image([])
                        stop_btn = st.button("Stop Tracking")
                        
                        while cap.isOpened() and not stop_btn:
                            ret, frame = cap.read()
                            if not ret: break
                            
                            # Convert to RGB for tracking update
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # Update Tracker
                            success, new_box = tracker_obj.update(frame_rgb)
                            
                            if success:
                                x, y, w, h = [int(v) for v in new_box]
                                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 3)
                                cv2.putText(frame_rgb, "Tracking", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                            else:
                                cv2.putText(frame_rgb, "Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                            
                            # Display (It is already RGB)
                            st_track_window.image(frame_rgb)
                            
                        cap.release()
            else:
                st.warning("Draw a box first!")

    # --- MODE 3: SAM2 DEMO ---
    elif track_mode == "3. SAM2 Segmentation Demo":
        st.subheader("SAM2 Segmentation Overlay")
        v_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
        n_file = st.file_uploader("Upload NPZ Mask File", type=['npz'])
        
        if v_file and n_file and st.button("Run SAM2 Demo"):
            with open("temp_video.mp4", "wb") as f: f.write(v_file.read())
            mask_data = np.load(n_file)
            cap = cv2.VideoCapture("temp_video.mp4")
            st_sam_window = st.image([])
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                output = tracker.overlay_sam2_mask(frame, frame_count, mask_data)
                st_sam_window.image(output, channels="BGR")
                frame_count += 1
                cv2.waitKey(30) 
            cap.release()

# --- MODULE 6: STEREO & POSE ESTIMATION ---
elif module_selection == "6. Stereo & Pose Estimation":
    st.header("6. Stereo Vision & Pose Estimation")
    
    mode_7 = st.radio("Select Task:", ["Task 1: Stereo Size Estimation", "Task 2: Pose & Hand Tracking"], horizontal=True)
    
    # TASK 1: STEREO MEASUREMENT
    if mode_7 == "Task 1: Stereo Size Estimation":
        st.subheader("Stereo Size Estimation")
        st.markdown("""
        *Instructions:*
        1. Provide a *Left* and *Right* image.
        2. Click *3 points* on each: Top-Left, Top-Right, Bottom-Left.
        """)
        
        f = stereo_pose.load_calibration("calibration_data_mac.npz")
        baseline = st.number_input("Baseline (Distance between cameras) in cm:", value=30.0)
        
        col1, col2 = st.columns(2)
        
        # LEFT IMAGE
        with col1:
            st.markdown("### 1. Left View")
            src_L = st.radio("Input Source:", ("Upload", "Camera"), key="src_L", horizontal=True)
            file_L = st.file_uploader("Upload Left", key="fL") if src_L == "Upload" else st.camera_input("Cam Left", key="cL")
            
            points_L = []
            if file_L:
                file_L.seek(0)
                img_L = cv2.imdecode(np.asarray(bytearray(file_L.read()), dtype=np.uint8), 1)
                img_L_rgb = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
                
                orig_h, orig_w = img_L.shape[:2]
                canvas_h = 350
                canvas_w = int(orig_w * (canvas_h / orig_h))
                scale_factor_L = orig_h / canvas_h

                st.write("*Click 3 Points (Top-L, Top-R, Bot-L):*")
                canvas_L = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.3)", stroke_color="#00FF00", stroke_width=4,
                    background_image=Image.fromarray(img_L_rgb),
                    height=canvas_h, width=canvas_w,
                    drawing_mode="point", key="canvas_L"
                )
                
                if canvas_L.json_data:
                    for obj in canvas_L.json_data["objects"]:
                        x_scaled = obj["left"] * scale_factor_L
                        y_scaled = obj["top"] * scale_factor_L
                        points_L.append((x_scaled, y_scaled))
                
                st.caption(f"Points: {len(points_L)}/3")

        # RIGHT IMAGE
        with col2:
            st.markdown("### 2. Right View")
            src_R = st.radio("Input Source:", ("Upload", "Camera"), key="src_R", horizontal=True)
            file_R = st.file_uploader("Upload Right", key="fR") if src_R == "Upload" else st.camera_input("Cam Right", key="cR")
            
            points_R = []
            if file_R:
                file_R.seek(0)
                img_R = cv2.imdecode(np.asarray(bytearray(file_R.read()), dtype=np.uint8), 1)
                img_R_rgb = cv2.cvtColor(img_R, cv2.COLOR_BGR2RGB)
                
                orig_h_R, orig_w_R = img_R.shape[:2]
                canvas_h = 350
                canvas_w_R = int(orig_w_R * (canvas_h / orig_h_R))
                scale_factor_R = orig_h_R / canvas_h

                st.write("*Click SAME 3 Points:*")
                canvas_R = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)", stroke_color="#FF0000", stroke_width=4,
                    background_image=Image.fromarray(img_R_rgb),
                    height=canvas_h, width=canvas_w_R,
                    drawing_mode="point", key="canvas_R"
                )
                
                if canvas_R.json_data:
                    for obj in canvas_R.json_data["objects"]:
                        x_scaled = obj["left"] * scale_factor_R
                        y_scaled = obj["top"] * scale_factor_R
                        points_R.append((x_scaled, y_scaled))
                
                st.caption(f"Points: {len(points_R)}/3")

        st.divider()
        if st.button("Calculate Real Dimensions", type="primary"):
            if len(points_L) == 3 and len(points_R) == 3:
                results, msg = stereo_pose.calculate_stereo_metrics(points_L, points_R, baseline, f)
                if results:
                    st.success("Calculation Successful!")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Avg Depth (Z)", f"{results['z_avg']} cm")
                    m2.metric("Real Width", f"{results['real_w']} cm")
                    m3.metric("Real Height", f"{results['real_h']} cm")
                else:
                    st.error(f"Calculation Failed: {msg}")
            else:
                st.warning(f"Selection Incomplete. Left: {len(points_L)}/3. Right: {len(points_R)}/3.")

