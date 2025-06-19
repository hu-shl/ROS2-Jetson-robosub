import cv2
import numpy as np
from collections import deque, Counter
import time

class LampAssistedColorDetector:
    def __init__(self):
        # Available buoy colors
        self.buoy_colors = ['red', 'white', 'black', 'orange', 'yellow']
        # Lamp colors for detection assistance
        self.lamp_colors = ['white', 'red', 'blue', 'green', 'off']
        # Current lamp state
        self.current_lamp_color = 'white'
        self.lamp_sequence_index = 0
        self.frames_per_lamp_color = 15  # Frames to keep each lamp color
        self.frame_counter = 0
        # Detection control
        self.detection_active = False
        self.detection_results = {}
        self.initialize_detection_system()
        # HSV ranges for each buoy color
        self.color_ranges = {
            'red': [(0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)],
            'white': [(0, 0, 180), (180, 55, 255)],
            'black': [(0, 0, 0), (180, 255, 45)],
            'orange': [(8, 150, 100), (25, 255, 255)],
            'yellow': [(22, 150, 100), (35, 255, 255)]
        }
        # Which lamp colors help detect which buoy colors best
        self.lamp_effectiveness = {
            'red': {'white': 1.0, 'blue': 0.3, 'green': 0.2, 'red': 0.1, 'off': 0.8},
            'white': {'blue': 1.0, 'red': 0.9, 'green': 0.8, 'white': 0.7, 'off': 0.3},
            'black': {'white': 1.0, 'red': 0.8, 'blue': 0.8, 'green': 0.7, 'off': 0.2},
            'orange': {'blue': 1.0, 'white': 0.9, 'green': 0.4, 'red': 0.3, 'off': 0.6},
            'yellow': {'blue': 1.0, 'white': 0.8, 'red': 0.3, 'green': 0.2, 'off': 0.5}
        }

    def initialize_detection_system(self):
        """Initialize the detection system without starting"""
        self.detection_results = {lamp_color: {} for lamp_color in self.lamp_colors}
        self.lamp_sequence_index = 0
        self.frame_counter = 0
        self.current_lamp_color = 'white'
        self.detection_active = False

    def start_detection_cycle(self):
        """Call this function to start a new detection cycle"""
        print("Starting new color detection cycle...")
        self.detection_active = True
        self.detection_results = {lamp_color: {} for lamp_color in self.lamp_colors}
        self.lamp_sequence_index = 0
        self.frame_counter = 0
        self.current_lamp_color = self.lamp_colors[0]
        self.set_physical_lamp_color(self.current_lamp_color)
        return True

    def stop_detection_cycle(self):
        """Stop the current detection cycle"""
        print("Stopping detection cycle...")
        self.detection_active = False
        self.set_physical_lamp_color('white')  # Return to default
        return self.get_final_color_guess()

    def analyze_color_with_current_lamp(self, frame, x, y, width, height):
        """Analyze color under current lamp conditions"""
        roi = frame[y:y+height, x:x+width]
        if roi.size == 0:
            return {}
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_scores = {}
        for color_name, ranges in self.color_ranges.items():
            if color_name == 'red':
                mask1 = cv2.inRange(hsv_roi, np.array(ranges[0]), np.array(ranges[1]))
                mask2 = cv2.inRange(hsv_roi, np.array(ranges[2]), np.array(ranges[3]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv_roi, np.array(ranges[0]), np.array(ranges[1]))
            color_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            raw_score = color_pixels / total_pixels if total_pixels > 0 else 0
            lamp_multiplier = self.lamp_effectiveness[color_name].get(self.current_lamp_color, 0.5)
            adjusted_score = raw_score * lamp_multiplier
            color_scores[color_name] = adjusted_score
        return color_scores

    def run_complete_detection_on_image(self, image_path, x1, y1, x2, y2):
        """Run complete detection cycle on a single image"""
        frame = cv2.imread(image_path)
        if frame is None:
            print("Failed to load image.")
            return None
        width = x2 - x1
        height = y2 - y1
        print(f"Running complete detection on region ({x1},{y1}) to ({x2},{y2})")
        all_results = {}
        for lamp_color in self.lamp_colors:
            print(f"Testing with {lamp_color} lamp...")
            self.current_lamp_color = lamp_color
            color_scores = self.analyze_color_with_current_lamp(frame, x1, y1, width, height)
            all_results[lamp_color] = color_scores
            print(f"  Scores: {color_scores}")
        self.detection_results = all_results
        final_color = self.get_final_color_guess()
        confidence = self.get_detection_confidence()
        print(f"\nFinal Result: {final_color} (confidence: {confidence:.2f})")
        return final_color, confidence

    def advance_lamp_sequence(self):
        """Move to next lamp color in sequence"""
        if self.lamp_sequence_index < len(self.lamp_colors) - 1:
            self.lamp_sequence_index += 1
            self.current_lamp_color = self.lamp_colors[self.lamp_sequence_index]
            self.set_physical_lamp_color(self.current_lamp_color)
            print(f"Lamp changed to: {self.current_lamp_color}")

    def get_current_best_guess(self):
        """Get best color guess based on all lamp tests so far"""
        if not any(self.detection_results.values()):
            return 'unknown'
        combined_scores = {}
        for color in self.buoy_colors:
            combined_scores[color] = 0
            weight_sum = 0
            for lamp_color, results in self.detection_results.items():
                if results:
                    score = results.get(color, 0)
                    weight = 1.0
                    combined_scores[color] += score * weight
                    weight_sum += weight
            if weight_sum > 0:
                combined_scores[color] /= weight_sum
        if combined_scores:
            best_color = max(combined_scores, key=combined_scores.get)
            if combined_scores[best_color] > 0.1:
                return best_color
        return 'unknown'

    def get_detection_confidence(self):
        """Calculate confidence in current detection"""
        best_guess = self.get_current_best_guess()
        if best_guess == 'unknown':
            return 0.0
        supporting_conditions = 0
        total_conditions = 0
        for lamp_color, results in self.detection_results.items():
            if results:
                total_conditions += 1
                color_scores = list(results.values())
                if results.get(best_guess, 0) == max(color_scores):
                    supporting_conditions += 1
        if total_conditions == 0:
            return 0.0
        return supporting_conditions / total_conditions

    def get_final_color_guess(self):
        """Get final answer after complete cycle"""
        best_guess = self.get_current_best_guess()
        if best_guess == 'unknown':
            return 'unknown'
        color_votes = {}
        for color in self.buoy_colors:
            votes = 0
            total_score = 0
            for lamp_color, results in self.detection_results.items():
                if results:
                    scores = list(results.values())
                    if scores and results.get(color, 0) == max(scores):
                        votes += 1
                    total_score += results.get(color, 0)
            color_votes[color] = {
                'votes': votes,
                'total_score': total_score,
                'combined': votes * 0.7 + total_score * 0.3
            }
        best_color = max(color_votes.keys(), key=lambda c: color_votes[c]['combined'])
        return best_color

    def set_physical_lamp_color(self, color):
        """Set the actual lamp color - implement based on your hardware"""
        print(f"Setting lamp to {color}")
        # IMPLEMENT YOUR LAMP CONTROL HERE

    def detect_color_without_lamp(self, frame, x, y, width, height):
        """
        Detect the buoy color in the given ROI without changing lamp color.
        Returns the best color guess and a confidence score.
        """
        roi = frame[y:y+height, x:x+width]
        if roi.size == 0:
            return 'unknown', 0.0
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_scores = {}
        for color_name, ranges in self.color_ranges.items():
            if color_name == 'red':
                mask1 = cv2.inRange(hsv_roi, np.array(ranges[0]), np.array(ranges[1]))
                mask2 = cv2.inRange(hsv_roi, np.array(ranges[2]), np.array(ranges[3]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv_roi, np.array(ranges[0]), np.array(ranges[1]))
            color_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            score = color_pixels / total_pixels if total_pixels > 0 else 0
            color_scores[color_name] = score
        best_color = max(color_scores, key=color_scores.get)
        confidence = color_scores[best_color]
        if confidence < 0.1:
            return 'unknown', confidence
        return best_color, confidence

# Usage Example

def example_single_image_analysis():
    """Analyze a single image with specified coordinates"""
    detector = LampAssistedColorDetector()
    image_path = "images/rami_marine_dataset/class_1/red/img_0012.png"
    x1, y1 = 500, 197  # Top-left corner
    x2, y2 = 816, 502  # Bottom-right corner
    result = detector.run_complete_detection_on_image(image_path, x1, y1, x2, y2)
    if result:
        final_color, confidence = result
        print(f"Detected color: {final_color} with {confidence:.1%} confidence")

def example_your_specific_use_case():
    """How to integrate with your existing code"""
    detector = LampAssistedColorDetector()
    buoy_found = True
    buoy_coordinates = (500, 197, 816, 502)  # x1, y1, x2, y2
    if buoy_found:
        x1, y1, x2, y2 = buoy_coordinates
        width = x2 - x1
        height = y2 - y1
        print("Buoy detected! Starting color identification...")
        detector.start_detection_cycle()
        frame = cv2.imread("images/rami_marine_dataset/class_1/red/img_0012.png")
        if frame is not None:
            result = detector.run_complete_detection_on_image("images/rami_marine_dataset/class_1/red/img_0012.png", x1, y1, x2, y2)
            if result:
                final_color, confidence = result
                print(f"Detection complete! Color: {final_color}")
                print(f"Confidence: {confidence:.1%}")

def example_no_lamp_analysis():
    """Analyze a single image without lamp assistance"""
    detector = LampAssistedColorDetector()
    image_path = "images/rami_marine_dataset/class_1/red/img_0012.png"
    x1, y1 = 500, 197  # Top-left corner
    x2, y2 = 816, 502  # Bottom-right corner
    frame = cv2.imread(image_path)
    if frame is not None:
        width = x2 - x1
        height = y2 - y1
        color, confidence = detector.detect_color_without_lamp(frame, x1, y1, width, height)
        print(f"[NO LAMP] Detected color: {color} with {confidence:.1%} confidence")
    else:
        print("Failed to load image.")

if __name__ == "__main__":
    print("Running Example: Single image analysis")
    example_single_image_analysis()
    print("\n" + "="*50)
    print("Running Example: Your specific use case simulation")
    example_your_specific_use_case()
    print("\n" + "="*50)
    print("Running Example: No lamp analysis")
    example_no_lamp_analysis()