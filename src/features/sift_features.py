'''
Helpers para extrair SIFT / Dense SIFT descriptors e utilitários de patch.
'''
import cv2 as cv
import numpy as np

class SIFTExtractor:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.type = cfg.get('type', 'sift')
        self.dense = cfg.get('dense', False)
        self.dense_step = int(cfg.get('dense_step', 8) or 8)
        self.sift = cv.SIFT_create()

    def extract_from_image(self, img_gray):
        if self.dense:
            return self.extract_dense(img_gray, step=self.dense_step)
        else:
            kps, des = self.sift.detectAndCompute(img_gray, None)
            return des if des is not None else np.zeros((0, 128), dtype=np.float32)

    def extract_from_path(self, path):
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Não foi possível ler {path}")
        return self.extract_from_image(img)

    def extract_dense(self, img_gray, step=8, patch_size=16):
        # generate grid of keypoints
        h, w = img_gray.shape[:2]
        kps = []
        for y in range(step//2, h, step):
            for x in range(step//2, w, step):
                kps.append(cv.KeyPoint(x, y, patch_size))
        kps, des = self.sift.compute(img_gray, kps)
        return des if des is not None else np.zeros((0, 128), dtype=np.float32)

# small utility to extract patch given kp (optional)
def patch_from_kp(img_gray, kp, size=32):
    x, y = int(kp.pt[0]), int(kp.pt[1])
    h, w = img_gray.shape[:2]
    half = size // 2
    x0, y0 = max(0, x-half), max(0, y-half)
    x1, y1 = min(w, x+half), min(h, y+half)
    return img_gray[y0:y1, x0:x1]
