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

qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

class OpiDetectionNode(Node):
    def __init__(self):
        super().__init__('opi_detection_node')
        self.publisher_ = self.create_publisher(OpiDetection, '/fmu/in/opi_detection', qos_profile)
        



    def on_detection(self, detections, frame=None):
        class_names = ["buoy", "1", "2", "3", "4"]
        img_dims = (AR0234_CONFIG['width'], AR0234_CONFIG['height'])
        color_detector = LampAssistedColorDetector()
        pipe_direction_detector = PipeDirectionDetector()
  

        for detection in detections:        
            confidence = detection['confidence']  # Detection confidence score (float, 0-1)
            if confidence < CONFIG['confidence_threshold']:
                continue  # Skip low-confidence detections            
            
            msg = OpiDetection()
            class_id = detection.get('class_id', 0)  # Index of detected class (int)
            class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)  # Class label (str)
            x1, y1, x2, y2 = detection.get('bbox', (0, 0, 0, 0))  # Bounding box coordinates (ints)
            distance = fast_distance_gpu(x1, y1, x2, y2)  # Estimated distance to object (float, cm)
            angles = calc_angle((x1, y1, x2, y2), img_dims)  # Dictionary with viewing angle info
            angle_x = angles.get('angle_x', 0)  # Horizontal angle (float, degrees)
            angle_y = angles.get('angle_y', 0)  # Vertical angle (float, degrees)
            detection['color'] = None  # Placeholder for color info
            detection['color_confidence'] = None  # Placeholder for color confidence
            direction_x,direction_y = angles.get('direction_x', 'direction_y')

            
            msg = OpiDetection()
            detection['color'] = None
            detection['color_confidence'] = None

            if class_name == "buoy":
                if distance <= 100 and frame is not None:
                    color, color_confidence = color_detector.detect_color_without_lamp(frame, x1, y1, x2, y2)
                    detection['color'] = color
                    detection['color_confidence'] = color_confidence                    
                    
                    msg.opi = msg.OPI_BOUY
                    msg.distance = int(distance)
                    msg.angle_hor = angle_x
                    msg.angle_ver = angle_y
                    msg.confidence = int(confidence)
                    msg.color = color
                    msg.timestamp = self.get_clock().now().nanoseconds // 1000
                    self.publisher_.publish(msg)
                
                else:
                    msg.opi = msg.OPI_BOUY
                    msg.distance = int(distance)
                    msg.angle_hor = angle_x
                    msg.angle_ver = angle_y
                    msg.confidence = int(confidence)
                    msg.color = color
                    msg.timestamp = self.get_clock().now().nanoseconds // 1000
                    self.publisher_.publish(msg)
            
            elif class_name in ["1", "2", "3", "4"]:
                # Assign unique OPI value for each marker
                marker_opi_map = {
                    "1": msg.OPI_ONE,
                    "2": msg.OPI_TWO,
                    "3": msg.OPI_THREE,
                    "4": msg.OPI_FOUR
                }
                
                msg.opi = marker_opi_map.get(class_name, msg.OPI_ONE)
                msg.distance = int(distance)
                msg.angle_hor = angle_x
                msg.angle_ver = angle_y
                msg.confidence = int(confidence)
                msg.timestamp = self.get_clock().now().nanoseconds // 1000
                if distance <= 100 and frame is not None:
                    direction, angle = pipe_direction_detector.detect_pipe_direction(frame)
                    msg.angle_pipe = angle
                self.publisher_.publish(msg)

            elif class_name in ["5", "6", "7", "8"]:
                #TODO: Implement detection for classes 5-8
                continue
            elif class_name == "red":
                #TODO: Implement detection for red color
                continue
            elif class_name == "green":
                #TODO: Implement detection for green color
                continue
            else:
                print(f"Detected {class_name} at {distance:.1f}cm, angle {angle_x:.1f}° {direction_x}")

    def autonomous_mode(self):
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
                    self.on_detection(result['all_detections'], frame)
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
    rclpy.init()
    node = OpiDetectionNode()
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
                node.autonomous_mode()
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
                node.autonomous_mode()
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
