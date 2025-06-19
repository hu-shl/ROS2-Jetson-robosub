import os
import time
import numpy as np
import torch
from ultralytics import YOLO
from queue import Queue, Empty
import threading
from concurrent.futures import ThreadPoolExecutor
from opi_detection.include.config import CONFIG
from opi_detection.include.image_processing import calc_angle
import cv2

CLASS_NAMES = ["buoy", "1", "2", "3", "4"]

def check_gpu_setup():
    print("🔍 GPU Setup Check:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.mm(test_tensor, test_tensor)
            print("✅ GPU tensor operations working")
        except Exception as e:
            print(f"⚠️ GPU test failed: {e}")
            CONFIG['device'] = 'cpu'
            CONFIG['half_precision'] = False
    else:
        print("❌ CUDA not available, falling back to CPU")
        CONFIG['device'] = 'cpu'
        CONFIG['half_precision'] = False
    print(f"Using device: {CONFIG['device']}")
    return CONFIG['device'] == 'cuda'

def draw_detections_gpu(frame, result):
    """GPU-optimized detection drawing"""
    if not result['success']:
        return frame

    colors = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 165, 255)]

    for i, detection in enumerate(result['all_detections']):
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']

        color = colors[i % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, CONFIG['bbox_thickness'])

        angles = calc_angle(detection['bbox'], result['img_dims'])

        # Only show color if guessed and only on buoys
        color_text = ""
        is_buoy = detection.get('class_id', 0) == 0 or detection.get('class_name', '') == "buoy"
        if is_buoy and detection.get('color') and detection['color'] != "unknown":
            color_text = f"Color: {detection['color']}"
            if detection.get('color_confidence') is not None:
                color_text += f" ({detection['color_confidence']:.0%})"

        texts = [
            f"Buoy #{i+1}: {detection['distance']:.1f}cm",
            f"Conf: {confidence:.3f}",
            f"H: {angles['angle_x']:.1f}° {angles['direction_x']}",
            f"V: {angles['angle_y']:.1f}° {angles['direction_y']}"
        ]
        if color_text:
            texts.insert(0, color_text)  # Show color at the top

        text_bg_height = len(texts) * 18 + 5
        cv2.rectangle(frame, (x1, y1-text_bg_height), (x1+220, y1), color, -1)

        for j, text in enumerate(texts):
            y_pos = max(y1 - text_bg_height + 15 + j * 18, 15)
            cv2.putText(frame, text, (x1+5, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX, CONFIG['text_font_scale'],
                      (0, 0, 0), CONFIG['text_thickness'])

    return frame

class GPUOptimizedDetector:
    """GPU-optimized detection class with TensorRT support"""
    def __init__(self):
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1)  # GPU inference is already parallel
        self.last_result = None
        self.inference_cache = {}
        self.cache_timeout = 0.05  # Reduced cache timeout for GPU speed
        self.gpu_available = check_gpu_setup()
        self.tensorrt_model = None
        
    def load_model(self):
        """Load model with GPU optimizations"""
        if self.model is None:
            if not os.path.exists(CONFIG['model_path']):
                print(f"❌ Model not found at {CONFIG['model_path']}")
                # Try alternative model paths
                alt_paths = ['yolov8n.pt', 'yolov8s.pt', 'best.pt']
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        CONFIG['model_path'] = alt_path
                        print(f"✅ Using alternative model: {alt_path}")
                        break
                else:
                    print("❌ No model found. Downloading YOLOv8n...")
                    CONFIG['model_path'] = 'yolov8n.pt'
            
            # Load model
            self.model = YOLO(CONFIG['model_path'])
            
            if self.gpu_available:
                print("🔥 Warming up GPU model...")
                
                # GPU warm-up with proper tensor placement
                dummy_img = torch.zeros((1, 3, CONFIG['input_size'], CONFIG['input_size']))
                if CONFIG['device'] == 'cuda':
                    dummy_img = dummy_img.cuda()
                    if CONFIG['half_precision']:
                        dummy_img = dummy_img.half()
                
                # Multiple warmup runs for GPU optimization
                for i in range(5):
                    start_time = time.time()
                    with torch.no_grad():
                        _ = self.model(dummy_img, 
                                     device=CONFIG['device'], 
                                     verbose=False,
                                     half=CONFIG['half_precision'])
                    warmup_time = time.time() - start_time
                    print(f"   GPU Warmup {i+1}/5: {warmup_time*1000:.1f}ms")
                
                # Try to convert to TensorRT for maximum speed
                if CONFIG['use_tensorrt']:
                    self.try_tensorrt_conversion()
                    
            else:
                # CPU warmup
                dummy_img = np.zeros((CONFIG['input_size'], CONFIG['input_size'], 3), dtype=np.uint8)
                self.model(dummy_img, device=CONFIG['device'], verbose=False)
                
            print(f"✅ Model loaded on {CONFIG['device']}")
        return True
    
    def try_tensorrt_conversion(self):
        """Try to convert model to TensorRT for maximum speed"""
        try:
            print("🚀 Attempting TensorRT conversion...")
            tensorrt_path = CONFIG['model_path'].replace('.pt', '.engine').replace('.onnx', '.engine')
            
            if os.path.exists(tensorrt_path):
                print(f"✅ Found existing TensorRT engine: {tensorrt_path}")
                self.tensorrt_model = YOLO(tensorrt_path)
                return True
            
            # Export to TensorRT
            print("⚙️ Converting to TensorRT (this may take a few minutes)...")
            self.model.export(
                format='engine',
                device=0,
                workspace=2,  # GB
                int8=False,  # Use FP16 instead for better compatibility
                half=True,
                dynamic=False,
                batch=1,
                imgsz=CONFIG['input_size']
            )
            
            if os.path.exists(tensorrt_path):
                self.tensorrt_model = YOLO(tensorrt_path)
                print("✅ TensorRT conversion successful!")
                return True
                
        except Exception as e:
            print(f"⚠️ TensorRT conversion failed: {e}")
            print("   Continuing with standard GPU inference...")
        
        return False
    
    def detect_async_gpu(self, image):
        """GPU-optimized async detection"""
        current_time = time.time()
        
        # Use TensorRT model if available, otherwise standard model
        active_model = self.tensorrt_model if self.tensorrt_model else self.model
        
        if active_model is None:
            return {'success': False}
        
        # Prepare image for GPU inference
        if CONFIG['device'] == 'cuda':
            # Ensure image is on GPU
            if isinstance(image, np.ndarray):
                # Convert numpy to tensor and move to GPU
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                image_tensor = image_tensor.unsqueeze(0).cuda()
                
                if CONFIG['half_precision']:
                    image_tensor = image_tensor.half()
            else:
                image_tensor = image
        else:
            image_tensor = image
        
        # Run inference with GPU optimizations
        with torch.no_grad():  # Disable gradient computation for speed
            results = active_model(
                image_tensor if CONFIG['device'] == 'cuda' else image,
                device=CONFIG['device'],
                verbose=False,
                conf=CONFIG['confidence_threshold'],
                iou=0.7,
                max_det=CONFIG['max_det'],
                agnostic_nms=CONFIG['agnostic_nms'],
                half=CONFIG['half_precision'],
                imgsz=CONFIG['input_size']
            )[0]
        
        if len(results.boxes) == 0:
            return {'success': False}
        
        img_height, img_width = image.shape[:2]
        all_detections = []
        
        # Process results on GPU then move to CPU for final processing
        for box in results.boxes:
            confidence = float(box.conf.cpu())  # Move to CPU for Python processing
            if confidence < CONFIG['confidence_threshold']:
                continue
                
            # Move coordinates to CPU for processing
            coords = box.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = map(int, coords)
            
            dist = self.fast_distance_gpu(x1, y1, x2, y2)
            
            all_detections.append({
                'distance': dist,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
            })
        
        if not all_detections:
            return {'success': False}
        
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'success': True,
            'img_dims': (img_width, img_height),
            'all_detections': all_detections
        }
    
    @staticmethod
    def fast_distance_gpu(x1, y1, x2, y2):
        """GPU-optimized distance calculation"""
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        longest_side = max(width, height)
        
        if longest_side <= 0:
            return 0
        
        return (CONFIG['focal_length'] * CONFIG['known_width']) / longest_side

class AsyncGPUDetector:
    """Fully asynchronous GPU detector"""
    def __init__(self):
        self.model = None
        self.tensorrt_model = None
        self.detection_thread = None
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.latest_result = None
        self.running = False
        self.inference_count = 0
        self.total_inference_time = 0
        self.gpu_available = check_gpu_setup()
        
        # GPU memory management
        if self.gpu_available:
            torch.cuda.empty_cache()
            # Reserve GPU memory
            dummy = torch.zeros((1, 3, CONFIG['input_size'], CONFIG['input_size'])).cuda()
            del dummy
            torch.cuda.empty_cache()
        
    def load_model(self):
        """Load model with GPU optimizations"""
        if self.model is None:
            model_path = CONFIG['model_path']
            
            # Check for model file
            if not os.path.exists(model_path):
                print(f"❌ Model not found at {model_path}")
                alt_paths = ['yolov8n.pt', 'yolov8s.pt', 'best.pt']
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        CONFIG['model_path'] = alt_path
                        print(f"✅ Using alternative model: {alt_path}")
                        break
                else:
                    print("📥 Downloading YOLOv8n...")
                    model_path = 'yolov8n.pt'
                    CONFIG['model_path'] = model_path
            
            # Load model
            self.model = YOLO(model_path)
            
            if self.gpu_available:
                print("🔥 GPU model warmup...")
                
                # Extensive GPU warmup
                dummy_img = np.zeros((CONFIG['input_size'], CONFIG['input_size'], 3), dtype=np.uint8)
                
                for i in range(3):
                    start_time = time.time()
                    with torch.no_grad():
                        _ = self.model(dummy_img, 
                                     device=CONFIG['device'], 
                                     verbose=False,
                                     half=CONFIG['half_precision'],
                                     imgsz=CONFIG['input_size'])
                    warmup_time = time.time() - start_time
                    print(f"   Warmup {i+1}/3: {warmup_time*1000:.1f}ms")
                
                # Try TensorRT conversion
                if CONFIG['use_tensorrt']:
                    self.setup_tensorrt()
                
                print(f"✅ GPU model ready on {CONFIG['device']}")
            else:
                # CPU warmup
                dummy_img = np.zeros((CONFIG['input_size'], CONFIG['input_size'], 3), dtype=np.uint8)
                self.model(dummy_img, device=CONFIG['device'], verbose=False)
                print("✅ CPU model ready")
                
        return True
    
    def setup_tensorrt(self):
        """Setup TensorRT engine"""
        try:
            tensorrt_path = CONFIG['model_path'].replace('.pt', '.engine').replace('.onnx', '.engine')
            
            if os.path.exists(tensorrt_path):
                print(f"✅ Loading existing TensorRT engine")
                self.tensorrt_model = YOLO(tensorrt_path)
                return
            
            print("⚙️ Creating TensorRT engine...")
            exported_path = self.model.export(
                format='engine',
                device=0,
                workspace=2,
                half=True,
                dynamic=False,
                batch=1,
                imgsz=CONFIG['input_size'],
                verbose=False
            )
            
            if os.path.exists(exported_path):
                self.tensorrt_model = YOLO(exported_path)
                print("✅ TensorRT engine created and loaded!")
            
        except Exception as e:
            print(f"⚠️ TensorRT setup failed: {e}")
    
    def start_detection_thread(self):
        """Start async detection thread"""
        self.running = True
        self.detection_thread = threading.Thread(target=self._gpu_detection_worker, daemon=True)
        self.detection_thread.start()
        print("🔄 GPU detection thread started")
    
    def _gpu_detection_worker(self):
        """GPU detection worker thread"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if frame is None:
                    continue
                
                start_time = time.time()
                result = self._fast_gpu_detect(frame)
                inference_time = time.time() - start_time
                
                self.inference_count += 1
                self.total_inference_time += inference_time
                
                # Store result
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except Empty:
                        break
                
                try:
                    self.result_queue.put_nowait(result)
                    self.latest_result = result
                except:
                    pass
                    
                # GPU memory cleanup
                if self.gpu_available and self.inference_count % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Empty:
                continue
            except Exception as e:
                print(f"⚠️ GPU detection error: {e}")
    
    def _fast_gpu_detect(self, image):
        """Ultra-fast GPU detection"""
        if self.model is None:
            return {'success': False, 'inference_time': 0}
        
        start_time = time.time()
        
        # Use TensorRT model if available
        active_model = self.tensorrt_model if self.tensorrt_model else self.model
        
        # GPU-optimized inference
        with torch.no_grad():
            if self.gpu_available:
                # Preprocess on GPU if possible
                results = active_model(image, 
                                     device=CONFIG['device'], 
                                     imgsz=CONFIG['input_size'],
                                     verbose=False,
                                     conf=CONFIG['confidence_threshold'],
                                     max_det=CONFIG['max_det'],
                                     agnostic_nms=CONFIG['agnostic_nms'],
                                     half=CONFIG['half_precision'])[0]
            else:
                results = active_model(image, 
                                     device=CONFIG['device'], 
                                     imgsz=CONFIG['input_size'],
                                     verbose=False,
                                     conf=CONFIG['confidence_threshold'],
                                     max_det=CONFIG['max_det'])[0]
        
        inference_time = time.time() - start_time
        
        if len(results.boxes) == 0:
            return {'success': False, 'inference_time': inference_time}
        
        # Process results
        original_height, original_width = image.shape[:2]
        all_detections = []
        
        for box in results.boxes:
            confidence = float(box.conf.cpu() if self.gpu_available else box.conf)
            if confidence < CONFIG['confidence_threshold']:
                continue
            
            coords = box.xyxy[0].cpu().tolist() if self.gpu_available else box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)
            
            # Fast distance calculation
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            longest_side = max(width, height)
            
            if longest_side > 0:
                dist = (CONFIG['focal_length'] * CONFIG['known_width']) / longest_side
            else:
                dist = 0
            
            all_detections.append({
                'distance': dist,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
            })
        
        if not all_detections:
            return {'success': False, 'inference_time': inference_time}
        
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        all_detections = all_detections[:CONFIG['max_det']]
        
        return {
            'success': True,
            'img_dims': (original_width, original_height),
            'all_detections': all_detections,
            'inference_time': inference_time
        }
    
    def submit_frame(self, frame):
        """Submit frame for async GPU detection"""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        try:
            self.frame_queue.put_nowait(frame.copy())
        except:
            pass
    
    def get_latest_result(self):
        """Get latest detection result"""
        try:
            while not self.result_queue.empty():
                self.latest_result = self.result_queue.get_nowait()
            return self.latest_result
        except Empty:
            return self.latest_result
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if self.inference_count > 0:
            avg_time = self.total_inference_time / self.inference_count
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            return {
                'avg_inference_time': avg_time,
                'avg_inference_fps': avg_fps,
                'total_inferences': self.inference_count
            }
        return {'avg_inference_time': 0, 'avg_inference_fps': 0, 'total_inferences': 0}
    
    def stop(self):
        """Stop detection thread"""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
        
        # GPU cleanup
        if self.gpu_available:
            torch.cuda.empty_cache()
