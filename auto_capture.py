import cv2
import os
import time

label = 'A'  # change this for each letter
person = 'b'
num_images = 250
delay = 0.09  # seconds between captures

save_dir = f'dataset/{label}'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
print("Press 's' to start capturing, 'q' to quit.")

started = False
count = 0
last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a box in the center
    h, w, _ = frame.shape
    box_size = 300
    x1, y1 = w//2 - box_size//2, h//2 - box_size//2
    x2, y2 = x1 + box_size, y1 + box_size
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Images Captured: {count}/{num_images}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Auto Capture - Place your hand inside the box", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        print("Started capturing...")
        started = True
        last_capture_time = time.time()

    if key == ord('q') or count >= num_images:
        break

    # Only start capturing after 's' is pressed
    if started and (time.time() - last_capture_time) >= delay:
        roi = frame[y1:y2, x1:x2]  # Extract ROI
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        roi_resized = cv2.resize(roi_gray, (64, 64))      # Resize
        cv2.imwrite(f"{save_dir}/{label}_be_{person}_{count}.jpg", roi_resized)
        count += 1
        last_capture_time = time.time()

print("Finished.")
cap.release()
cv2.destroyAllWindows()
