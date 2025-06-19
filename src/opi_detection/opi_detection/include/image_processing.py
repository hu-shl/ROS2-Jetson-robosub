import cv2
import numpy as np
import time
from opi_detection.include.config import AR0234_CONFIG, CONFIG
from opi_detection.include.camera_control import update_exposure

def calculate_brightness(frame):
    """Calculate frame brightness"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def convert_bayer_to_rgb(frame):
    """Enhanced Bayer to RGB conversion"""
    try:
        if len(frame.shape) == 3:
            bayer_data = frame[:, :, 0]
        else:
            bayer_data = frame

        if bayer_data.dtype == np.uint16:
            bayer_8bit = (bayer_data >> 2).astype(np.uint8)
        else:
            bayer_8bit = bayer_data.astype(np.uint8)

        try:
            rgb_frame = cv2.cvtColor(bayer_8bit, cv2.COLOR_BAYER_GR2RGB)
        except:
            if len(frame.shape) == 3:
                return frame
            else:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Apply color gains
        if (AR0234_CONFIG['red_gain'] != 1.0 or 
            AR0234_CONFIG['green_gain'] != 1.0 or 
            AR0234_CONFIG['blue_gain'] != 1.0):
            
            rgb_frame = rgb_frame.astype(np.float32)
            rgb_frame[:,:,0] *= AR0234_CONFIG['blue_gain']
            rgb_frame[:,:,1] *= AR0234_CONFIG['green_gain']
            rgb_frame[:,:,2] *= AR0234_CONFIG['red_gain']
            rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)
        
        return rgb_frame
        
    except Exception as e:
        print(f"⚠️ Color conversion failed: {e}")
        if len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

def convert_bayer_to_rgb_fast(frame):
    """Ultra-fast Bayer conversion for display"""
    try:
        if len(frame.shape) == 3:
            return frame  # Already RGB
        
        # Simple demosaicing - much faster
        if frame.dtype == np.uint16:
            frame = (frame >> 4).astype(np.uint8)  # Convert 12-bit to 8-bit
        
        # Use simple interpolation instead of full demosaicing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_GR2BGR)
        return rgb_frame
        
    except:
        if len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

def auto_adjust_exposure_gpu(frame, cap):
    """GPU-compatible non-blocking exposure adjustment check"""
    if not CONFIG['auto_exposure_enabled']:
        return
        
    brightness = calculate_brightness(frame)
    target_min = CONFIG['target_brightness_min']
    target_max = CONFIG['target_brightness_max']
    
    # Only queue adjustments for extreme cases
    if brightness < target_min - 20:  # Very dark
        print(f"📷 Brightness too low: {brightness:.1f} (will adjust when possible)")
    elif brightness > target_max + 20:  # Very bright
        print(f"📷 Brightness too high: {brightness:.1f} (will adjust when possible)")

def adjust_exposure_to_target(cap, target_brightness, max_iterations=8):
    """Auto-adjust exposure to reach target brightness with improved algorithm"""
    print(f"🎯 Auto-adjusting exposure to target brightness: {target_brightness}")
    
    for iteration in range(max_iterations):
        ret, frame = cap.read()
        if not ret:
            return None
            
        frame = convert_bayer_to_rgb(frame)
        current_brightness = calculate_brightness(frame)
        print(f"   Iteration {iteration+1}: brightness = {current_brightness:.1f}")
        
        # Check if we're close enough
        if abs(current_brightness - target_brightness) < 5:
            print(f"   ✅ Target reached! Final brightness: {current_brightness:.1f}")
            return frame
        
        # Calculate adjustment factor with improved logic
        brightness_ratio = target_brightness / current_brightness
        
        if current_brightness < target_brightness:
            # Too dark - increase exposure (but be conservative)
            adjustment_factor = min(1.3, max(1.1, brightness_ratio * 0.8))
        else:
            # Too bright - decrease exposure
            adjustment_factor = max(0.7, min(0.9, brightness_ratio * 1.2))
    
        # Apply adjustment
        current_exposure = AR0234_CONFIG['exposure']
        new_exposure = int(current_exposure * adjustment_factor)
        new_exposure = max(50, min(30000, new_exposure))  # Clamp limits
        
        AR0234_CONFIG['exposure'] = new_exposure
        update_exposure()
        print(f"   📷 Exposure adjusted: {current_exposure} → {new_exposure} (factor: {adjustment_factor:.2f})")
        
        # Stabilization frames - let camera adjust naturally
        stabilization_frames = 5
        for frame_num in range(stabilization_frames):
            time.sleep(0.15)
            ret, stabilize_frame = cap.read()
            if ret:
                stabilize_frame = convert_bayer_to_rgb(stabilize_frame)
                frame_brightness = calculate_brightness(stabilize_frame)
                if frame_num == stabilization_frames - 1:  # Last frame
                    print(f"      Stabilized brightness: {frame_brightness:.1f}")

    # Final capture after all adjustments
    ret, final_frame = cap.read()
    if ret:
        final_frame = convert_bayer_to_rgb(final_frame)
        final_brightness = calculate_brightness(final_frame)
        print(f"   ⚠️  Max iterations reached. Final brightness: {final_brightness:.1f}")
        return final_frame
    
    return None

def auto_adjust_white_balance_gpu(frame):
    """GPU-compatible white balance adjustment using unified logic"""
    if not CONFIG['auto_wb_enabled']:
        return

    # Only adjust every few calls to prevent constant changes
    if hasattr(auto_adjust_white_balance_gpu, 'call_count'):
        auto_adjust_white_balance_gpu.call_count += 1
    else:
        auto_adjust_white_balance_gpu.call_count = 0

    if auto_adjust_white_balance_gpu.call_count % 10 != 0:
        return

    # Calculate new white balance gains using unified function
    gains = auto_white_balance(frame)

    # Apply gains with smoothing to prevent abrupt changes
    smooth_factor = 0.1  # Gentle changes for stability

    AR0234_CONFIG['red_gain'] = AR0234_CONFIG['red_gain'] * (1 - smooth_factor) + gains['red_gain'] * smooth_factor
    AR0234_CONFIG['green_gain'] = AR0234_CONFIG['green_gain'] * (1 - smooth_factor) + gains['green_gain'] * smooth_factor
    AR0234_CONFIG['blue_gain'] = AR0234_CONFIG['blue_gain'] * (1 - smooth_factor) + gains['blue_gain'] * smooth_factor

    # Round for cleaner display
    AR0234_CONFIG['red_gain'] = round(AR0234_CONFIG['red_gain'], 2)
    AR0234_CONFIG['green_gain'] = round(AR0234_CONFIG['green_gain'], 2)
    AR0234_CONFIG['blue_gain'] = round(AR0234_CONFIG['blue_gain'], 2)

def calc_angle(bbox, img_dims):
    """Calculate viewing angles"""
    x1, y1, x2, y2 = bbox
    img_width, img_height = img_dims
    
    midx = x1 + ((x2 - x1)/2)
    midy = y1 + ((y2 - y1)/2)
    
    rel_x = (midx - img_width/2) / (img_width/2)
    rel_y = (midy - img_height/2) / (img_height/2)
    
    anglex = rel_x * (CONFIG['camera_fov_h'] / 2)
    angley = rel_y * (CONFIG['camera_fov_v'] / 2)
    
    direction_x = "right" if anglex > 0 else "left"
    direction_y = "down" if angley > 0 else "up"
    
    return {
        'angle_x': abs(anglex),
        'angle_y': abs(angley),
        'direction_x': direction_x,
        'direction_y': direction_y
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

def auto_white_balance(frame):
    """
    Unified white balance adjustment.
    - Uses brightest 10% of pixels for smart WB.
    - Falls back to conservative WB if not enough bright pixels.
    Returns: dict with 'red_gain', 'green_gain', 'blue_gain'
    """
    try:
        # Smart method: LAB color space, brightest 10%
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)
        brightness_threshold = np.percentile(l, 90)
        bright_mask = l > brightness_threshold

        if np.sum(bright_mask) >= 100:
            b_ch, g, r = cv2.split(frame)
            r_bright = np.mean(r[bright_mask])
            g_bright = np.mean(g[bright_mask])
            b_bright = np.mean(b_ch[bright_mask])
            target = (r_bright + g_bright + b_bright) / 3

            r_gain = target / r_bright if r_bright > 0 else 1.0
            g_gain = target / g_bright if g_bright > 0 else 1.0
            b_gain = target / b_bright if b_bright > 0 else 1.0

            # Conservative limits
            r_gain = np.clip(r_gain, 0.6, 1.6)
            g_gain = np.clip(g_gain, 0.6, 1.6)
            b_gain = np.clip(b_gain, 0.6, 1.6)
        else:
            # Fallback: conservative method
            b, g, r = cv2.split(frame)
            mask = (r > 20) & (g > 20) & (b > 20) & (r < 235) & (g < 235) & (b < 235)
            if np.sum(mask) < 1000:
                return {'red_gain': 1.0, 'green_gain': 1.0, 'blue_gain': 1.0}
            r_avg = np.mean(r[mask])
            g_avg = np.mean(g[mask])
            b_avg = np.mean(b[mask])
            target = g_avg
            r_gain = target / r_avg if r_avg > 0 else 1.0
            g_gain = 1.0
            b_gain = target / b_avg if b_avg > 0 else 1.0
            r_gain = np.clip(r_gain, 0.7, 1.3)
            b_gain = np.clip(b_gain, 0.7, 1.3)

        return {
            'red_gain': round(r_gain, 2),
            'green_gain': round(g_gain, 2),
            'blue_gain': round(b_gain, 2)
        }
    except Exception as e:
        print(f"⚠️  White balance failed: {e}")
        return {'red_gain': 1.0, 'green_gain': 1.0, 'blue_gain': 1.0}