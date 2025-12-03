import numpy as np
import cv2


# -------------------------------------------------------------------
# 1. Gaussian PSF
# -------------------------------------------------------------------

def _gaussian_kernel(ksize=51, sigma=15.0):
    """
    Generate a normalized 2D Gaussian kernel (PSF).
    """
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


# -------------------------------------------------------------------
# 2. Wiener deconvolution with padding (to avoid wrap-around artifacts)
# -------------------------------------------------------------------

def _wiener_deconvolution(blurred, kernel, K=0.01):
    """
    Wiener deconvolution in frequency domain for a single channel,
    with image padding to avoid circular wrap-around artifacts.
    blurred : 2D uint8 array
    kernel  : 2D PSF (small Gaussian kernel)
    K       : noise-to-signal power ratio
    """
    # Normalize to [0,1]
    g = blurred.astype(np.float32) / 255.0
    H, W = g.shape

    # 1) Pad image (2x size)
    pad_h, pad_w = H * 2, W * 2
    g_pad = np.zeros((pad_h, pad_w), np.float32)
    g_pad[:H, :W] = g  # place in top-left

    # 2) Pad kernel to same size
    kernel_pad = np.zeros((pad_h, pad_w), np.float32)
    kh, kw = kernel.shape
    kernel_pad[:kh, :kw] = kernel

    # 3) FFTs
    G = np.fft.fft2(g_pad)
    Hf = np.fft.fft2(kernel_pad)

    # 4) Wiener filter
    H_conj = np.conj(Hf)
    denom = (np.abs(Hf) ** 2) + K
    F_est = (H_conj / denom) * G

    # 5) Inverse FFT and crop
    f_est = np.fft.ifft2(F_est)
    f_est = np.real(f_est)

    # crop back to original size
    f_crop = f_est[:H, :W]

    # 6) Clip and convert to uint8
    f_crop = np.clip(f_crop, 0, 1)
    f_crop = (f_crop * 255.0).astype(np.uint8)

    return f_crop


# -------------------------------------------------------------------
# 3. Function called by app.py
# -------------------------------------------------------------------

def apply_blur_and_restore(original_bgr):
    """
    Input:
        original_bgr: BGR image from Streamlit uploader.
    Output:
        blurred_bgr:  heavily blurred version
        restored_bgr: restored using Fourier/Wiener deconvolution
    """
    if original_bgr is None:
        return None, None

    # Strong blur parameters (you can tweak these)
    ksize = 31        # large kernel for obvious blur
    sigma = 8.0       # strong Gaussian blur
    K = 0.004        # Wiener regularization (increase for smoother, decrease for sharper/noisier)

    kernel = _gaussian_kernel(ksize=ksize, sigma=sigma)

    # Split channels
    b, g, r = cv2.split(original_bgr)

    # --- 1. Apply blur in spatial domain ---
    b_blur = cv2.filter2D(b, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    g_blur = cv2.filter2D(g, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    r_blur = cv2.filter2D(r, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    blurred_bgr = cv2.merge([b_blur, g_blur, r_blur])

    # --- 2. Restore each channel using Wiener deconvolution ---
    b_rec = _wiener_deconvolution(b_blur, kernel, K=K)
    g_rec = _wiener_deconvolution(g_blur, kernel, K=K)
    r_rec = _wiener_deconvolution(r_blur, kernel, K=K)

    restored_bgr = cv2.merge([b_rec, g_rec, r_rec])

    return blurred_bgr, restored_bgr
