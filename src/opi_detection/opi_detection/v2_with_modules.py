#!/usr/bin/env python3
import cv2
import torch
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from px4_msgs.msg import OpiDetection

from opi_detection.include.config import AR0234_CONFIG, CONFIG
from opi_detection.include.camera_control import set_camera_controls, update_exposure, update_gain
from opi_detection.include.image_processing import (
    convert_bayer_to_rgb,
    calculate_brightness,
    auto_adjust_exposure_gpu,
    auto_adjust_white_balance_gpu,
    calc_angle,
    fast_distance_gpu
)
from opi_detection.include.detector import (
    check_gpu_setup,
    GPUOptimizedDetector,
    AsyncGPUDetector,
    draw_detections_gpu,
)
from opi_detection.include.color_Detection import LampAssistedColorDetector
from opi_detection.include.edge import PipeDirectionDetector  # Add this import

tmpmsg = OpiDetection()

def on_detection(detections, frame=None):
    class_names = ["buoy", "1", "2", "3", "4"]
    img_dims = (AR0234_CONFIG['width'], AR0234_CONFIG['height'])
    color_detector = LampAssistedColorDetector()
    pipe_direction_detector = PipeDirectionDetector()  

    for detection in detections:
        confidence = detection.get('confidence', 0)
        if confidence < CONFIG['confidence_threshold']:
            continue  # Skip low-confidence detections
        class_id = detection.get('class_id', 0)
        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        x1, y1, x2, y2 = detection.get('bbox', (0, 0, 0, 0))
        distance = fast_distance_gpu(x1, y1, x2, y2)
        angles = calc_angle((x1, y1, x2, y2), img_dims)
        angle_x = angles.get('angle_x', 0)
        direction_x = angles.get('direction_x', '')
        detection['color'] = None
        detection['color_confidence'] = None

        if class_name == "buoy":
            print(f"🌊 Buoy detected at {distance:.1f}cm, angle {angle_x:.1f}° {direction_x}")
            if distance <= 100 and frame is not None:
                width = x2 - x1
                height = y2 - y1
                color, confidence = color_detector.detect_color_without_lamp(frame, int(x1), int(y1), int(width), int(height))
                print(f"Buoy color detected: {color} (confidence: {confidence:.1%})")
                detection['color'] = color
                detection['color_confidence'] = confidence
                print(f"bouy detected at {distance:.1f}cm, angle {angle_x:.1f}° {direction_x} with color {color} (confidence: {confidence:.1%})")
            else:
                print(f"bouy detected at 100 cm < {distance:.1f}cm, angle {angle_x:.1f}° {direction_x}")
        
        elif class_name in ["1", "2", "3", "4"]:
            print(f"Marker {class_name} detected at {distance:.1f}cm, angle {angle_x:.1f}° {direction_x}")
            if frame is not None:
                direction, angle = pipe_direction_detector.detect_pipe_direction(frame)
                print(f"  Pipe direction: {direction}, Angle: {angle}")
        
        else:
            print(f"Detected {class_name} at {distance:.1f}cm, angle {angle_x:.1f}° {direction_x}")

def autonomous_mode():
    CONFIG['auto_exposure_enabled'] = True
    CONFIG['auto_wb_enabled'] = True
    detector = AsyncGPUDetector()
    if not detector.load_model():
        print("❌ Failed to load model")
        return
    detector.start_detection_thread()
    cap = cv2.VideoCapture(CONFIG['camera_device'])
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    set_camera_controls()
    time.sleep(1.0)
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0
    auto_adjust_counter = 0
    result = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame_count += 1
            if frame_count % CONFIG['inference_interval'] == 0:
                frame = convert_bayer_to_rgb(frame)
                auto_adjust_counter += 1
                if auto_adjust_counter >= 120:
                    auto_adjust_counter = 0
                    auto_adjust_exposure_gpu(frame, cap)
                    auto_adjust_white_balance_gpu(frame)
            else:
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame_count % CONFIG['inference_interval'] == 0:
                detector.submit_frame(frame)
            latest_result = detector.get_latest_result()
            if latest_result is not None:
                result = latest_result
            if result is not None and result.get('success', False):
                frame = draw_detections_gpu(frame, result)
            if result and result.get('all_detections'):
                on_detection(result['all_detections'], frame)
            cv2.imshow(CONFIG['window_name'], frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                break
            elif key == ord('w'):
                print("🟦 Smart white balance triggered by user")
                auto_adjust_white_balance_gpu(frame)
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()
        cap.release()
        cv2.destroyAllWindows()
        if CONFIG['device'] == 'cuda':
            torch.cuda.empty_cache()

def main():
    print("🔍 GPU-Optimized AR0234 Detection System")
    print("=" * 50)
    gpu_available = check_gpu_setup()
    if gpu_available:
        print("\n🚀 GPU detected and working!")
        print("Choose mode:")
        print("1. GPU-Optimized Detection (recommended)")
        print("2. Performance Test")
        print("3. Settings Only")
        print("4. Autonomous Mode")
    else:
        print("\n⚠️ GPU not available, using CPU mode")
        print("1. CPU Detection")
        print("2. Settings Only")
        print("3. Autonomous Mode")
    try:
        if gpu_available:
            choice = input("Enter choice (1-4, default=1): ").strip()
            if choice == "2":
                pass  # test_gpu_performance() if implemented in include
            elif choice == "3":
                cap = cv2.VideoCapture(CONFIG['camera_device'])
                if cap.isOpened():
                    set_camera_controls()
                    # simple_settings_mode(cap) if implemented in include
                    cap.release()
                    cv2.destroyAllWindows()
            elif choice == "4":
                autonomous_mode()
            else:
                pass  # gpu_optimized_main() if implemented in include
        else:
            choice = input("Enter choice (1-3, default=1): ").strip()
            if choice == "2":
                cap = cv2.VideoCapture(CONFIG['camera_device'])
                if cap.isOpened():
                    set_camera_controls()
                    # simple_settings_mode(cap) if implemented in include
                    cap.release()
                    cv2.destroyAllWindows()
            elif choice == "3":
                autonomous_mode()
            else:
                CONFIG['device'] = 'cpu'
                CONFIG['half_precision'] = False
                pass  # gpu_optimized_main() if implemented in include
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()