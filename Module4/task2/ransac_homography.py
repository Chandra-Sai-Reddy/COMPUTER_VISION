import numpy as np

def compute_homography(src_pts, dst_pts):
    # src_pts, dst_pts: (N, 2)
    N = src_pts.shape[0]
    A = []
    for i in range(N):
        x, y = src_pts[i]
        xp, yp = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])

    A = np.array(A, dtype=np.float32)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    return H


def ransac_homography(src_pts, dst_pts, num_iter=1000, thresh=3.0):
    best_H = None
    best_inliers = []

    N = src_pts.shape[0]
    if N < 4:
        return None, np.array([])

    for _ in range(num_iter):
        idx = np.random.choice(N, 4, replace=False)
        H_candidate = compute_homography(src_pts[idx], dst_pts[idx])

        # Project all src_pts
        src_h = np.hstack([src_pts, np.ones((N, 1))])
        proj = (H_candidate @ src_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]

        errors = np.linalg.norm(proj - dst_pts, axis=1)
        inliers = np.where(errors < thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H_candidate

    return best_H, best_inliers
