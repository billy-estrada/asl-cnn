import cv2
import os
import time
import string

person = 'dark'
images_per_letter = 40
delay_between_images = 0.18  # seconds between each image
delay_between_letters = 3.5  # seconds between letters (non-blocking)

# Letters A to Y excluding 'J'
labels = [ch for ch in string.ascii_uppercase if ch != 'J']

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Camera not available.")

print("Press 's' to start or 'q' to quit.")
started = False

# Initial wait loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    box_size = 300
    x1, y1 = w // 2 - box_size // 2, h // 2 - box_size // 2
    x2, y2 = x1 + box_size, y1 + box_size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Press 's' to start A-Y capture (no J)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Capture Setup", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        started = True
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

if started:
    for i, label in enumerate(labels):
        save_dir = f'dataset/{label}'
        os.makedirs(save_dir, exist_ok=True)
        next_label = labels[i+1] if i+1 < len(labels) else None
        count = 0
        last_capture_time = time.time()

        # Capture 10 images for current letter
        while count < images_per_letter:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            x1, y1 = w // 2 - 150, h // 2 - 150
            x2, y2 = x1 + 300, y1 + 300

            current_time = time.time()

            if current_time - last_capture_time >= delay_between_images:
                roi = frame[y1:y2, x1:x2]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_resized = cv2.resize(roi_gray, (64, 64))
                filename = f"{save_dir}/{label}_be_{person}_{count}.jpg"
                cv2.imwrite(filename, roi_resized)
                count += 1
                last_capture_time = current_time

            # Draw overlays
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Letter: {label}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            if next_label:
                cv2.putText(frame, f"Next: {next_label}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            cv2.imshow("Capturing", frame)
            cv2.waitKey(1)

        # Show "Next" message for 2 seconds with live feed
        start_next_time = time.time()
        while time.time() - start_next_time < delay_between_letters:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, f"Next: {next_label if next_label else 'Done'}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
            cv2.imshow("Capturing", frame)
            cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
