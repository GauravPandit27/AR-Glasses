import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Function to rotate an image
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Rotate around the center
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated

# Function to overlay the transparent image
def overlay_transparent(background, overlay, x, y):
    bg = background.copy()
    h, w = overlay.shape[:2]

    if x + w > bg.shape[1] or y + h > bg.shape[0] or x < 0 or y < 0:
        return bg

    # If the glasses image doesn't have an alpha channel, we create one
    if overlay.shape[2] == 3:
        # Create a fully opaque alpha channel
        alpha_channel = np.ones((h, w), dtype=np.uint8) * 255
        overlay = np.dstack((overlay, alpha_channel))  # Add alpha channel to glasses

    alpha = overlay[:, :, 3] / 255.0  # Extract alpha channel
    for c in range(3):
        bg[y:y+h, x:x+w, c] = alpha * overlay[:, :, c] + (1 - alpha) * bg[y:y+h, x:x+w, c]
    return bg

# Load multiple glasses PNGs with alpha channel (ensure they have transparency)
glasses_folder = 'glasses'  # Folder where your glasses PNGs are stored
glasses_files = [f for f in os.listdir(glasses_folder) if f.endswith('.png')]
glasses_images = [cv2.imread(os.path.join(glasses_folder, f), cv2.IMREAD_UNCHANGED) for f in glasses_files]

# Start with the first pair of glasses
current_glasses_index = 0

# Create a fullscreen window
cv2.namedWindow("Face Mesh with Tilted Glasses", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Face Mesh with Tilted Glasses", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame_flipped = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape

            # Landmark points for left and right eye
            left = face_landmarks.landmark[33]  # Left eye (inner corner)
            right = face_landmarks.landmark[263]  # Right eye (inner corner)

            left_x, left_y = int(left.x * w), int(left.y * h)
            right_x, right_y = int(right.x * w), int(right.y * h)

            # Calculate center and angle between eyes
            center_x = (left_x + right_x) // 2
            center_y = (left_y + right_y) // 2
            dx = right_x - left_x
            dy = right_y - left_y
            angle = np.degrees(np.arctan2(dy, dx))  # Angle between eyes in degrees

            # Adjust angle back due to flipping for mirror effect
            angle = -angle

            # Calculate size for glasses based on distance between eyes
            eye_dist = np.linalg.norm([dx, dy])
            desired_width = int(eye_dist * 1.6)  # Adjust width relative to eye distance

            # Preserve original aspect ratio of the glasses image
            aspect_ratio = glasses_images[current_glasses_index].shape[1] / glasses_images[current_glasses_index].shape[0]
            desired_height = int(desired_width / aspect_ratio)

            # Resize the glasses image to fit over the eyes with correct aspect ratio
            glasses_resized = cv2.resize(glasses_images[current_glasses_index], (desired_width, desired_height))

            # Rotate the glasses to match the angle between the eyes
            rotated_glasses = rotate_image(glasses_resized, angle)

            # Position the glasses image around the center of the face
            y_offset = int(center_y - rotated_glasses.shape[0] / 2)
            x_offset = int(center_x - rotated_glasses.shape[1] / 2)

            # Overlay the rotated glasses image on the frame
            frame_flipped = overlay_transparent(frame_flipped, rotated_glasses, x_offset, y_offset)

            # Draw dimensions and metrics on top-right corner
            # Eye-to-eye distance and other face-related metrics
            text = f"Eye Distance: {eye_dist:.2f} px"
            cv2.putText(frame_flipped, text, (frame_flipped.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Calculate and display additional metrics like face width (distance between the leftmost and rightmost points on the face)
            leftmost = face_landmarks.landmark[234]  # Leftmost point on the face
            rightmost = face_landmarks.landmark[454]  # Rightmost point on the face

            leftmost_x, leftmost_y = int(leftmost.x * w), int(leftmost.y * h)
            rightmost_x, rightmost_y = int(rightmost.x * w), int(rightmost.y * h)

            face_width = np.linalg.norm([rightmost_x - leftmost_x, rightmost_y - leftmost_y])
            face_height = np.linalg.norm([right_y - left_y, right_x - left_x])  # Calculate height from eyes

            # Draw face dimensions
            cv2.putText(frame_flipped, f"Face Width: {face_width:.2f} px", (frame_flipped.shape[1] - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_flipped, f"Face Height: {face_height:.2f} px", (frame_flipped.shape[1] - 300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the final frame
    cv2.imshow("Face Mesh with Tilted Glasses", frame_flipped)

    # Wait for key press to cycle through glasses
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC to exit
        break
    elif key == ord('d'):  # 'd' to go forward to the next glasses
        current_glasses_index = (current_glasses_index + 1) % len(glasses_images)
    elif key == ord('a'):  # 'a' to go backward to the previous glasses
        current_glasses_index = (current_glasses_index - 1) % len(glasses_images)

cap.release()
cv2.destroyAllWindows()

