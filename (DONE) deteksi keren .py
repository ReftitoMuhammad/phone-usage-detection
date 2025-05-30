from datetime import datetime
from ultralytics import YOLO
import cv2
import time
import os
import pygame  # Import pygame for audio playback

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Load alarm sound file
alarm_sound_file = "alert.mp3"  # Replace with your actual sound file path
if os.path.exists(alarm_sound_file):
    alarm_sound = pygame.mixer.Sound(alarm_sound_file)
else:
    print(f"Warning: Alarm sound file '{alarm_sound_file}' not found.")
    alarm_sound = None

# Load YOLO models
model_phone = YOLO("Model/handphone.pt")
model_detect = YOLO("Model/HandphoneDetect_Fix.pt")

# Open camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise Exception("Cannot open camera")

# Separate states for phone detection and overlap warning
handphone_detected = False  # For general phone detection
overlap_detected = False    # For overlap warning/alarm
alarm_playing = False
start_time = None
overlap_start_time = None
driver_id = "Mntp123"
log_file = 'log/log_pelanggaran.txt'

def log_pelanggaran(start, end, duration):
    with open(log_file, 'a') as f:
        f.write(f"Event : Terdeteksi memegang handphone (Overlap)\n")
        f.write(f"Dari jam : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}\n")
        f.write(f"Sampai jam : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}\n")
        f.write(f"Durasi : {duration:.2f} detik\n\n")

def play_alarm():
    global alarm_playing
    if alarm_sound and not alarm_playing:
        alarm_sound.play(-1)  # Play on loop (-1)
        alarm_playing = True
        print("⚠️ ALARM: Phone overlap detected! ⚠️")

def stop_alarm():
    global alarm_playing
    if alarm_sound and alarm_playing:
        alarm_sound.stop()
        alarm_playing = False
        print("Alarm stopped")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    current_time = time.time()
    results_detect = model_detect.predict(frame, show=False)
    results_phone = model_phone.predict(frame, show=False)
    
    handphone_found = False
    overlap_found = False
    
    # First, detect all phones and draw them
    phone_boxes = []
    for result_phone in results_phone:
        for box_phone in result_phone.boxes:
            if box_phone.conf[0] > 0.10:
                x1_phone, y1_phone, x2_phone, y2_phone = map(int, box_phone.xyxy[0])
                confidence_phone = box_phone.conf[0]
                label_phone = f"Handphone {confidence_phone:.2f}"
                
                # Always draw phone detection (cyan/yellow color)
                cv2.rectangle(frame, (x1_phone, y1_phone), (x2_phone, y2_phone), (255, 255, 0), 2)
                cv2.putText(frame, label_phone, (x1_phone, y1_phone - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                phone_boxes.append((x1_phone, y1_phone, x2_phone, y2_phone))
                handphone_found = True
    
    # Then, check for hand/phone holding detection and overlap
    for result in results_detect:
        for box_detect in result.boxes:
            if box_detect.conf[0] > 0.40:
                class_detect = result.names[int(box_detect.cls)]
                confidence_detect = box_detect.conf[0]
                
                if class_detect in ['ga_nelpon', 'nelpon']:
                    x1_detect, y1_detect, x2_detect, y2_detect = map(int, box_detect.xyxy[0])
                    
                    # Check for overlap with any detected phone
                    has_overlap = False
                    for x1_phone, y1_phone, x2_phone, y2_phone in phone_boxes:
                        # Check if phone box is inside hand detection box (overlap)
                        if (x1_phone >= x1_detect and y1_phone >= y1_detect and 
                            x2_phone <= x2_detect and y2_phone <= y2_detect):
                            has_overlap = True
                            break
                    
                    if has_overlap:
                        # Overlap detected - show warning
                        overlap_found = True
                        label = f"PERINGATAN: Memegang HP {confidence_detect:.2f}"
                        cv2.rectangle(frame, (x1_detect, y1_detect), (x2_detect, y2_detect), (0, 0, 255), 2)  # Red for alarmko
                        cv2.putText(frame, label, (x1_detect, y1_detect - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        # Add warning text to the frame
                        warning_text = "WARNING: Phone overlap detected!"
                        cv2.putText(frame, warning_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        # No overlap - just show normal detection
                        # label = f"Deteksi Tangan {confidence_detect:.2f}"
                        # cv2.rectangle(frame, (x1_detect, y1_detect), (x2_detect, y2_detect), (0, 255, 0), 2)  # Green for normal
                        # cv2.putText(frame, label, (x1_detect, y1_detect - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        pass
    
    # Handle general phone detection state
    if handphone_found:
        if not handphone_detected:
            start_time = current_time
            handphone_detected = True
            print("📱 Phone detected in frame")
    else:
        if handphone_detected:
            handphone_detected = False
            print("📱 Phone no longer detected")
    
    # Handle overlap detection and alarm
    if overlap_found:
        if not overlap_detected:
            overlap_start_time = current_time
            overlap_detected = True
            play_alarm()  # Start alarm when overlap is detected
    else:
        if overlap_detected:
            # Overlap ended - log the violation and stop alarm
            end_time = current_time
            duration = end_time - overlap_start_time
            log_pelanggaran(overlap_start_time, end_time, duration)
            overlap_detected = False
            stop_alarm()  # Stop alarm when overlap is no longer detected
    
    # Show phone detection status (without alarm)
    if handphone_found and not overlap_found:
        status_text = "Phone detected (No overlap)"
        cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.imshow('Deteksi Handphone', frame)
    if cv2.waitKey(1) == ord('q'):
        if alarm_playing:
            stop_alarm()
        break

cam.release()
cv2.destroyAllWindows()
pygame.quit()  # Clean up pygame resources