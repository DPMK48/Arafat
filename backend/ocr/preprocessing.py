import cv2
import numpy as np
import time
from pathlib import Path
from PIL import Image

class PreprocessorConfig:
    def __init__(
        self,
        enable_deskew=True,
        denoise_method="bilateral",      # 'bilateral', 'gaussian', 'median', 'none'
        contrast_method="clahe",         # 'clahe', 'linear', 'none'
        clahe_clip_limit=2.5,
        clahe_grid_size=(8, 8),
        binarization_method="adaptive_gaussian", # 'adaptive_gaussian', 'otsu', 'none'
        morphology_repair=True           # connect broken characters & ruling lines
    ):
        self.enable_deskew = enable_deskew
        self.denoise_method = denoise_method
        self.contrast_method = contrast_method
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.binarization_method = binarization_method
        self.morphology_repair = morphology_repair

    def to_dict(self):
        return {
            "enable_deskew": self.enable_deskew,
            "denoise_method": self.denoise_method,
            "contrast_method": self.contrast_method,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_grid_size": list(self.clahe_grid_size),
            "binarization_method": self.binarization_method,
            "morphology_repair": self.morphology_repair
        }

class LegacyImagePreprocessor:
    """
    OpenCV-based image preprocessing pipeline designed for legacy Nigerian university academic records.
    Handles skew, faded typewriter text, coffee/paper stains, uneven illumination, and degraded ruling lines.
    """
    def __init__(self, default_config: PreprocessorConfig = None):
        self.config = default_config or PreprocessorConfig()

    def load_image(self, image_input):
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not load image at {image_input}")
            return img
        elif isinstance(image_input, np.ndarray):
            return image_input.copy()
        elif isinstance(image_input, Image.Image):
            return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            raise TypeError("Unsupported image input type")

    def validate_resolution(self, img, min_dim=600):
        h, w = img.shape[:2]
        if h < min_dim or w < min_dim:
            # Scale up if scanned at very low resolution
            scale = max(min_dim / h, min_dim / w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return img

    def detect_and_deskew(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        # Invert for contour analysis
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Filter small noise
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return img, 0.0
            
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # If angle is negligible, do not rotate
        if abs(angle) < 0.2 or abs(angle) > 40.0:
            return img, 0.0

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed, round(float(angle), 2)

    def denoise(self, gray, method="bilateral"):
        if method == "bilateral":
            return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        elif method == "gaussian":
            return cv2.GaussianBlur(gray, (3, 3), 0)
        elif method == "median":
            return cv2.medianBlur(gray, 3)
        return gray

    def enhance_contrast(self, gray, method="clahe", clip_limit=2.5, grid_size=(8, 8)):
        if method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            return clahe.apply(gray)
        elif method == "linear":
            return cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return gray

    def binarize(self, gray, method="adaptive_gaussian"):
        if method == "adaptive_gaussian":
            # Handles non-uniform illumination and scanner shadow gradients
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 11
            )
        elif method == "otsu":
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return bin_img
        return gray

    def repair_morphology(self, binary_img):
        # Dilate lightly to repair faded typewriter breaks then close
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        repaired = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        return repaired

    def process(self, image_input, custom_config: PreprocessorConfig = None, save_path=None):
        start_time = time.time()
        cfg = custom_config or self.config

        raw_img = self.load_image(image_input)
        img = self.validate_resolution(raw_img)

        skew_angle = 0.0
        if cfg.enable_deskew:
            img, skew_angle = self.detect_and_deskew(img)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        # Denoising
        denoised = self.denoise(gray, method=cfg.denoise_method)

        # Contrast enhancement for faded text
        enhanced = self.enhance_contrast(
            denoised, 
            method=cfg.contrast_method, 
            clip_limit=cfg.clahe_clip_limit, 
            grid_size=cfg.clahe_grid_size
        )

        # Binarization
        binarized = self.binarize(enhanced, method=cfg.binarization_method)

        # Morphological repair
        final_img = binarized
        if cfg.morphology_repair:
            final_img = self.repair_morphology(binarized)

        # Convert back to 3-channel for downstream layout / display
        output_rgb = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)

        elapsed = time.time() - start_time

        if save_path:
            cv2.imwrite(str(save_path), final_img)

        metadata = {
            "processing_time_sec": round(elapsed, 4),
            "skew_angle_deg": skew_angle,
            "original_dimensions": [raw_img.shape[1], raw_img.shape[0]],
            "processed_dimensions": [output_rgb.shape[1], output_rgb.shape[0]],
            "config_used": cfg.to_dict()
        }

        return output_rgb, metadata
