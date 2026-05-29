import cv2
import numpy as np
import math
import os
import sys
from glob import glob

class PipeDirectionDetector:
    def __init__(self):
        self.last_result = None  # Store the last output

    def preprocess_image(self, image):
        processed = cv2.bilateralFilter(image, 9, 75, 75)
        processed = self.underwater_color_correction(processed)
        return processed

    def underwater_color_correction(self, img):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(result)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        result = cv2.merge((l,a,b))
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    def enhanced_detect_pipe_structure(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Crop out top 15% to remove surface noise
        h, w = gray.shape
        crop_height = int(h * 0.15)
        gray_cropped = gray[crop_height:, :]
        
        # Apply smoothing and calculate gradients
        smoothed = cv2.GaussianBlur(gray_cropped, (15, 15), 4.0)
        grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=5)
        
        gradient_mag = cv2.magnitude(grad_x, grad_y)
        gradient_dir = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        
        # Get strong gradients and clean them up
        threshold_value = np.percentile(gradient_mag, 90)
        _, strong_gradients = cv2.threshold(gradient_mag, threshold_value, 255, cv2.THRESH_BINARY)
        strong_gradients = np.uint8(strong_gradients)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        strong_gradients = cv2.morphologyEx(strong_gradients, cv2.MORPH_CLOSE, kernel)
        strong_gradients = cv2.morphologyEx(strong_gradients, cv2.MORPH_OPEN, kernel)
        
        # Analyze edge directions
        edge_analysis = self.analyze_edge_directions(gradient_mag, gradient_dir, strong_gradients)
        
        return gradient_mag, gradient_dir, strong_gradients, crop_height, edge_analysis

    def analyze_edge_directions(self, gradient_mag, gradient_dir, strong_gradients):
        """Analyze edge directions to find pipe orientation"""
        strong_pixels = np.where(strong_gradients > 0)
        
        if len(strong_pixels[0]) == 0:
            return None
        
        # Get edge directions (perpendicular to gradients)
        edge_directions = gradient_dir[strong_pixels]
        edge_magnitudes = gradient_mag[strong_pixels]
        edge_orientations = (edge_directions + 90) % 180
        
        # Create weighted histogram
        weights = edge_magnitudes / np.max(edge_magnitudes)
        hist_bins = np.arange(0, 181, 10)
        weighted_hist, bin_edges = np.histogram(edge_orientations, bins=hist_bins, weights=weights)
        count_hist, _ = np.histogram(edge_orientations, bins=hist_bins)
        
        # Find dominant orientations
        edge_clusters = []
        for i in range(len(weighted_hist)):
            if count_hist[i] >= 10:
                bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                edge_clusters.append({
                    'orientation': bin_center,
                    'count': count_hist[i],
                    'weight': weighted_hist[i]
                })
        
        if not edge_clusters:
            return None
        
        edge_clusters.sort(key=lambda x: x['weight'], reverse=True)
        dominant_edge = edge_clusters[0]
        pipe_orientation = dominant_edge['orientation']
        
        direction_name = self.get_direction_name(pipe_orientation)
        
        return {
            'pipe_orientation': pipe_orientation,
            'direction_name': direction_name
        }

    def detect_pipe_direction(self, image):
        processed = self.preprocess_image(image)
        gradient_mag, gradient_dir, strong_gradients, crop_offset, edge_analysis = self.enhanced_detect_pipe_structure(processed)
        edge_analysis = self.analyze_edge_directions(gradient_mag, gradient_dir, strong_gradients)
        if edge_analysis:
            result = (edge_analysis['direction_name'], edge_analysis['pipe_orientation'])
        else:
            result = (None, None)
        previous_result = self.last_result
        self.last_result = result
        return result

    def compare_last_result(self):
        """Compare the last two results. Returns None if not enough data."""
        # This method assumes you call detect_pipe_direction at least twice
        # To compare, you need to store the previous result as well
        # Let's store both last and previous
        if not hasattr(self, 'previous_result'):
            self.previous_result = None
        if self.previous_result is None or self.last_result is None:
            return None
        comparison = {
            "previous": self.previous_result,
            "current": self.last_result,
            "changed": self.previous_result != self.last_result
        }
        return comparison

    def detect_and_compare(self, image):
        """Detect direction and compare with previous result."""
        if not hasattr(self, 'previous_result'):
            self.previous_result = None
        result = self.detect_pipe_direction(image)
        comparison = None
        if self.previous_result is not None:
            comparison = {
                "previous": self.previous_result,
                "current": result,
                "changed": self.previous_result != result
            }
        self.previous_result = result
        return result, comparison

def process_image(image_path, output_dir):
    """Process single image with both methods"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return
    
    basename = os.path.basename(image_path)
    print(f"\nProcessing {basename}")
    
    detector = PipeDirectionDetector()
    direction_name, pipe_orientation = detector.detect_pipe_direction(img)
    
    if direction_name is not None:
        print(f"✓ RESULT: {direction_name} ({pipe_orientation:.1f}°)")
    else:
        print("❌ NO RESULT")

def main():
    if len(sys.argv) < 3:
        print("Usage: python edge.py <image_or_folder> <output_folder>")
        return
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isdir(input_path):
        images = glob(os.path.join(input_path, "*.jpg")) + glob(os.path.join(input_path, "*.png"))
        print(f"Found {len(images)} images")
        for img_path in images:
            process_image(img_path, output_dir)
    else:
        process_image(input_path, output_dir)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()