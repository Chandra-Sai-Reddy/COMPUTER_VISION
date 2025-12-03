# module1_single_view.py

import streamlit as st
import subprocess
import os
import tempfile

def run():
    st.title("Module 1 – Single-View Size Estimation (Original Code)")

    st.markdown("""
    This module runs your **original measure.py file EXACTLY as it is**.
    No changes were made to your logic, clicking, or calculations.

    **Your OpenCV window will pop up exactly like before.**
    """)

    calib = st.file_uploader("Upload calibration_data_mac.npz", type=["npz"])
    img = st.file_uploader("Upload object.jpg", type=["jpg", "jpeg", "png"])

    if calib and img:
        # Temporary working folder
        temp_dir = tempfile.mkdtemp()

        calib_path = os.path.join(temp_dir, "calibration_data_mac.npz")
        img_path = os.path.join(temp_dir, "object.jpg")
        script_path = os.path.join(temp_dir, "measure.py")

        # Save uploaded files
        with open(calib_path, "wb") as f:
            f.write(calib.read())

        with open(img_path, "wb") as f:
            f.write(img.read())

        # Copy original measure.py as-is
        st.markdown("### Upload your original measure.py file:")
        measure = st.file_uploader("measure.py", type=["py"])

        if measure:
            with open(script_path, "wb") as f:
                f.write(measure.read())

            st.success("Files ready. Click below to run your EXACT script.")

            if st.button("Run measure.py"):
                st.info("Running your script… watch for an OpenCV window!")

                # Run unmodified code
                process = subprocess.Popen(
                    ["python3", script_path],
                    cwd=temp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                stdout, stderr = process.communicate()

                st.markdown("### Script Output (Console)")
                st.code(stdout)

                if stderr.strip():
                    st.markdown("### Script Errors")
                    st.code(stderr)
