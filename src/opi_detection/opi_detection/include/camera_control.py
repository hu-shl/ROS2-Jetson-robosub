import subprocess
import time
from opi_detection.include.config import AR0234_CONFIG, CONFIG

def set_camera_controls():
    """Set AR0234 camera controls using v4l2-ctl"""
    device = CONFIG['camera_device']
    
    resolution_cmd = f"v4l2-ctl -d {device} --set-fmt-video=width={AR0234_CONFIG['width']},height={AR0234_CONFIG['height']},pixelformat=RGGB"
    subprocess.run(resolution_cmd, shell=True, capture_output=True)
    time.sleep(0.5)
    
    verify_cmd = f"v4l2-ctl -d {device} --get-fmt-video"
    result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)
    print(f"Camera format:\n{result.stdout.strip()}")
    
    controls = [
        f"exposure={AR0234_CONFIG['exposure']}",
        f"analogue_gain={AR0234_CONFIG['analogue_gain']}",
        f"digital_gain={int(AR0234_CONFIG['digital_gain'] * 100)}",
    ]
    
    for ctrl in controls:
        cmd = f"v4l2-ctl -d {device} --set-ctrl={ctrl}"
        subprocess.run(cmd, shell=True, capture_output=True)
        time.sleep(0.1)
    
    AR0234_CONFIG['total_gain'] = AR0234_CONFIG['analogue_gain'] * AR0234_CONFIG['digital_gain']
    print(f"✅ Camera controls applied")

def update_exposure():
    cmd = f"v4l2-ctl -d {CONFIG['camera_device']} --set-ctrl=exposure={AR0234_CONFIG['exposure']}"
    for i in range(3):
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0 and i == 2:
            print(f"⚠️ Exposure update warning: {result.stderr.strip()}")
        time.sleep(0.05)

def update_gain():
    """Update camera gain"""
    cmd = f"v4l2-ctl -d {CONFIG['camera_device']} --set-ctrl=analogue_gain={AR0234_CONFIG['analogue_gain']}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ Gain update warning: {result.stderr.strip()}")
    
    AR0234_CONFIG['total_gain'] = AR0234_CONFIG['analogue_gain'] * AR0234_CONFIG['digital_gain']
