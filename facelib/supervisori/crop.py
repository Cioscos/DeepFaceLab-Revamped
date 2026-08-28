"""Crop a 5 punti verso il template ArcFace 112² (protocollo InsightFace `norm_crop`)."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Il template a 5 punti di ArcFace a 112x112 (insightface `arcface_dst`).
TEMPLATE_ARCFACE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def cinque_punti(landmark68):
    lm = np.asarray(landmark68, np.float32)
    if lm.shape != (68, 2):
        raise ValueError(f"servono 68 landmark, trovati {lm.shape}")
    return np.stack([lm[36:42].mean(0), lm[42:48].mean(0), lm[30], lm[48], lm[54]]).astype(np.float32)


def _matrice(punti5):
    M, _ = cv2.estimateAffinePartial2D(punti5, TEMPLATE_ARCFACE, method=cv2.LMEDS)
    if M is None:
        raise ValueError("similarita' verso il template non stimabile")
    return M.astype(np.float32)


def crop_112(immagine, landmark68):
    M = _matrice(cinque_punti(landmark68))
    return cv2.warpAffine(immagine.astype(np.float32), M, (112, 112), borderValue=0.0)


def matrice_fissa(punti5, lato):
    """theta (2x3) per `F.affine_grid(align_corners=False)`: porta il frame
    allineato lato x lato al template 112 con la stessa similarita' di
    `crop_112`. affine_grid vuole la mappa *inversa* (uscita -> ingresso) in
    coordinate normalizzate [-1, 1], con il centro del pixel i a (2i+1)/W - 1."""
    M = _matrice(np.asarray(punti5, np.float32))                    # pixel in -> pixel out
    M3 = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64)
    if not np.isfinite(M3).all() or abs(np.linalg.det(M3)) < 1e-9:
        raise ValueError("similarita' degenere")
    M_inv = np.linalg.inv(M3)                                       # pixel out -> pixel in
    a_in = np.array([[2.0 / lato, 0.0, 1.0 / lato - 1.0],
                     [0.0, 2.0 / lato, 1.0 / lato - 1.0],
                     [0.0, 0.0, 1.0]])                              # pixel in -> norm in
    a_out = np.array([[56.0, 0.0, 55.5], [0.0, 56.0, 55.5], [0.0, 0.0, 1.0]])  # norm out -> pixel out
    theta = a_in @ M_inv @ a_out
    return torch.tensor(theta[:2], dtype=torch.float32)


def crop_112_torch(x, theta):
    """x NCHW qualunque dtype -> N x C x 112 x 112 in float32, differenziabile in x."""
    x = x.float()
    t = theta.to(x.device, torch.float32).unsqueeze(0).expand(x.shape[0], 2, 3)
    griglia = F.affine_grid(t, (x.shape[0], x.shape[1], 112, 112), align_corners=False)
    return F.grid_sample(x, griglia, mode="bilinear", padding_mode="zeros", align_corners=False)
