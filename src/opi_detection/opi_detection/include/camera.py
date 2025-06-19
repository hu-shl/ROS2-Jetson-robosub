import cv2
from opi_detection.include.camera_control import set_camera_controls, update_exposure, update_gain
from opi_detection.include.image_processing import convert_bayer_to_rgb, calculate_brightness, smart_white_balance
from opi_detection.include.detector import GPUOptimizedDetector, AsyncGPUDetector
from opi_detection.include.config import AR0234_CONFIG, CONFIG

class Camera:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        set_camera_controls()

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image")
        return frame

    def release(self):
        self.cap.release()