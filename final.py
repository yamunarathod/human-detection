import cv2
import numpy as np
import threading
import time
import random
import math
from collections import deque
import textwrap

# Import pyk4a for Azure Kinect access
import pyk4a
from pyk4a import PyK4A, Config

# Load YOLO model
try:
    net = cv2.dnn.readNet("./yolo_model/yolov3.weights", "./yolo_model/yolov3.cfg")
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Random thoughts collection
RANDOM_THOUGHTS = [
    "Wow I look great!",
    "I wonder what's for lunch today.",
    "I can't wait to invest in Cisco's AI tech!",
    "AI-Ready Data Centers are my favorite kind of data centers!",
    "There is no defense like Cisco AI-Defense!",
    "Mirror, mirror on the wall, who's the smartest AI of all? (My bet's on Cisco AI)",
    "I am reimagining my data center for the AI era, with Cisco",
    "Power, speed, and security—while balancing costs? That's Cisco AI-Ready Datacenters",
    "Scaling for AI fast? I count on Cisco",
    "For built-in security and resilience, I choose Cisco.",
    "Evolving business needs? I trust Cisco to keep up.",
    "Complexity is everywhere. Cisco keeps it simple for me.",
    "I hope there's free Wi-Fi…and snacks.",
    "If I nod, will they think I understand everything?",
    "I wonder if anyone will notice if I sneak out for a nap.",
    "Maybe the AI can network for me while I find snacks",
    "AI brings new risks, but with Cisco AI Defense, I feel covered from end to end",
    "With so many third-party AI apps popping up, I'm glad Cisco can spot them before I do.",
    "I want to innovate with AI, not worry about security—and Cisco makes that possible",
    "I'd rather prevent a data leak than clean one up. Cisco AI Defense gets it.",
    "Real-time guardrails for AI? That's peace of mind, Thank You Cisco.",
    "With Cisco, I get to focus on AI innovation, not AI anxiety.",
    "Should I ask a question, or just nod like I already know the answer?",
    "How many cups of coffee is too many at a tech event?",
    "I'm here for the innovation, but I'll stay for the pastries."
]

class Person:
    def __init__(self, person_id, box, confidence):
        self.id = person_id
        self.box = box
        self.confidence = confidence
        self.thought = random.choice(RANDOM_THOUGHTS)
        self.thought_timer = 0
        self.bubble_animation = 0
        self.last_seen = time.time()
        self.prediction_box = box
        self.velocity = (0, 0)
        self.missed_frames = 0

    def update(self, box, confidence):
        old_center = (self.box[0] + self.box[2]//2, self.box[1] + self.box[3]//2)
        new_center = (box[0] + box[2]//2, box[1] + box[3]//2)
        self.velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])

        # Even more responsive smoothing for better tracking
        alpha = 0.7  # Reduced from 0.8 for even faster response
        self.box = [
            int(alpha * box[0] + (1 - alpha) * self.box[0]),
            int(alpha * box[1] + (1 - alpha) * self.box[1]),
            int(alpha * box[2] + (1 - alpha) * self.box[2]),
            int(alpha * box[3] + (1 - alpha) * self.box[3])
        ]

        self.confidence = confidence
        self.last_seen = time.time()
        self.thought_timer += 1
        self.bubble_animation += 0.15
        self.missed_frames = 0  # Reset missed frames when detected

        if self.thought_timer > 120:
            self.thought = random.choice(RANDOM_THOUGHTS)
            self.thought_timer = 0

    def predict_position(self):
        if self.missed_frames > 0:
            predicted_x = self.box[0] + self.velocity[0] * self.missed_frames
            predicted_y = self.box[1] + self.velocity[1] * self.missed_frames

            # Less aggressive damping for better prediction
            damping = 0.95 ** self.missed_frames  # Increased from 0.9
            self.velocity = (self.velocity[0] * damping, self.velocity[1] * damping)

            return [int(predicted_x), int(predicted_y), self.box[2], self.box[3]]
        return self.box

def calculate_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def wrap_text(text, max_chars_per_line=25):
    """Wrap text to multiple lines if it's too long"""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line + " " + word) <= max_chars_per_line:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines

def draw_oval_thought_bubble(frame, center_x, center_y, width, height, scale_factor=1.0):
    """Draw oval thought bubble matching the CSS oval design"""
    # Scale dimensions
    width = int(width * scale_factor)
    height = int(height * scale_factor)

    # Calculate bubble position (center)
    bubble_center_x = center_x
    bubble_center_y = center_y

    # Draw main oval bubble (white fill)
    cv2.ellipse(frame, (bubble_center_x, bubble_center_y), (width//2, height//2), 0, 0, 360, (255, 255, 255), -1)

    # Add subtle shadow/border for depth
    cv2.ellipse(frame, (bubble_center_x, bubble_center_y), (width//2, height//2), 0, 0, 360, (200, 200, 200), 2)

    # Draw thought bubble tail (two bigger circles)
    # First smaller circle - made bigger
    tail_size_1 = int(18 * scale_factor)  # Increased from 12
    tail_x_1 = bubble_center_x - width//6  # Adjusted position
    tail_y_1 = bubble_center_y + height//2 + int(8 * scale_factor)  # Reduced from 25
    cv2.circle(frame, (tail_x_1, tail_y_1), tail_size_1//2, (255, 255, 255), -1)
    cv2.circle(frame, (tail_x_1, tail_y_1), tail_size_1//2, (200, 200, 200), 1)

    # Second smaller circle - made bigger
    tail_size_2 = int(12 * scale_factor)  # Increased from 8
    tail_x_2 = bubble_center_x - width//4  # Adjusted position
    tail_y_2 = bubble_center_y + height//2 + int(16 * scale_factor)  # Reduced from 45
    cv2.circle(frame, (tail_x_2, tail_y_2), tail_size_2//2, (255, 255, 255), -1)
    cv2.circle(frame, (tail_x_2, tail_y_2), tail_size_2//2, (200, 200, 200), 1)

def draw_thought_bubble(frame, x, y, text, animation_offset=0, scale_factor=1.0):
    """Draw oval thought bubble with text - non-bold, larger font, better spacing"""

    # Text properties - increased font size and removed bold
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55 * scale_factor  # Increased from 0.45 for larger text
    font_thickness = 1  # Always 1 for non-bold text
    line_spacing = int(16 * scale_factor)  # Increased from 12 for better line height

    # Adjusted padding for larger text
    padding_x = int(18 * scale_factor)  # Slightly increased from 15
    padding_y = int(12 * scale_factor)  # Slightly increased from 10

    # Wrap text with more characters per line for smaller bubbles
    text_lines = wrap_text(text, max_chars_per_line=25)

    # Calculate text dimensions
    max_text_width = 0
    total_text_height = 0

    for i, line in enumerate(text_lines):
        (line_width, line_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        max_text_width = max(max_text_width, line_width)
        if i == 0:
            total_text_height += line_height
        else:
            total_text_height += line_spacing

    # Calculate oval bubble dimensions - adjusted for larger text
    bubble_width = max_text_width + (padding_x * 2)
    bubble_height = total_text_height + (padding_y * 2)

    # Adjusted constraints for larger text
    min_width = int(70 * scale_factor)   # Slightly increased from 60
    max_width = int(200 * scale_factor)  # Slightly increased from 180
    min_height = int(40 * scale_factor)  # Slightly increased from 35
    max_height = int(90 * scale_factor)  # Slightly increased from 80

    bubble_width = max(min_width, min(bubble_width, max_width))
    bubble_height = max(min_height, min(bubble_height, max_height))

    # Animation - smaller floating effect
    animation_y = int(2 * math.sin(animation_offset) * scale_factor)
    bubble_center_x = x
    bubble_center_y = y - bubble_height//2 - int(25 * scale_factor) + animation_y

    # Keep within frame bounds
    bubble_center_x = max(bubble_width//2 + 10, min(bubble_center_x, frame.shape[1] - bubble_width//2 - 10))
    bubble_center_y = max(bubble_height//2 + 20, bubble_center_y)

    # Draw the oval thought bubble
    draw_oval_thought_bubble(frame, bubble_center_x, bubble_center_y, bubble_width, bubble_height, scale_factor)

    # Draw text centered in the oval
    if len(text_lines) == 1:
        line = text_lines[0]
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        text_x = bubble_center_x - text_width // 2
        text_y = bubble_center_y + text_height // 2
        cv2.putText(frame, line, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
    else:
        start_y = bubble_center_y - (total_text_height // 2)
        current_y = start_y

        for i, line in enumerate(text_lines):
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            text_x = bubble_center_x - text_width // 2

            if i == 0:
                text_y = current_y + text_height
                current_y = text_y
            else:
                text_y = current_y + line_spacing
                current_y = text_y

            cv2.putText(frame, line, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

class PersonDetector:
    def __init__(self):
        self.net = net
        self.detections = []
        self.processing = False
        self.people = []
        self.next_person_id = 0
        self.detection_lock = threading.Lock()

    def detect_people(self, frame):
        if self.processing:
            return

        self.processing = True

        try:
            (height, width) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)

            output_layer_names = self.net.getUnconnectedOutLayersNames()
            output_layers = self.net.forward(output_layer_names)

            boxes = []
            confidences = []

            for output in output_layers:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if class_id == 0 and confidence > 0.35:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))

            current_detections = []
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.35, 0.25)
                if len(indices) > 0:
                    current_detections = [(boxes[i], confidences[i]) for i in indices.flatten()]

            with self.detection_lock:
                self.update_people_tracking(current_detections)

        except Exception as e:
            print(f"Detection error: {e}")
        finally:
            self.processing = False

    def update_people_tracking(self, detections):
        current_time = time.time()

        # Increment missed frames for all people
        for person in self.people:
            person.missed_frames += 1

        # Remove people who have been missed for too many frames - even faster cleanup
        self.people = [p for p in self.people if p.missed_frames < 8]  # Reduced from 15

        matched_people = set()

        for box, confidence in detections:
            best_overlap = 0
            best_person = None

            for i, person in enumerate(self.people):
                if i in matched_people:
                    continue

                compare_box = person.predict_position() if person.missed_frames > 0 else person.box
                overlap = calculate_overlap(box, compare_box)

                # More relaxed overlap threshold for better tracking
                if overlap > 0.15 and overlap > best_overlap:  # Reduced from 0.25
                    best_overlap = overlap
                    best_person = person

            if best_person and best_overlap > 0.15:  # Reduced from 0.25
                best_person.update(box, confidence)
                matched_people.add(self.people.index(best_person))
            else:
                new_person = Person(self.next_person_id, box, confidence)
                self.people.append(new_person)
                self.next_person_id += 1

        # Remove people who haven't been seen for too long - even faster removal
        self.people = [p for p in self.people if current_time - p.last_seen < 0.5]  # Reduced from 1.0

# Initialize Azure Kinect camera
k4a = None
actual_width = 1280 # Default if not retrieved
actual_height = 720 # Default if not retrieved

try:
    # Define the configuration
    k4a_config = Config(
        color_resolution=pyk4a.ColorResolution.RES_720P, # or RES_1080P, RES_1440P, etc.
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED, # or WFOV_2X2BINNED, etc.
        camera_fps=pyk4a.FPS.FPS_30 # or FPS_15, FPS_5
    )
    k4a = PyK4A(k4a_config) # Pass the config object directly
    k4a.start()
    print("Azure Kinect camera opened successfully!")

    # Get actual resolution only after the camera has started and a capture is available
    # It's better to get the dimensions from the first actual frame
    capture = k4a.get_capture()
    if capture.color is not None:
        actual_height, actual_width, _ = capture.color.shape
        print(f"Camera resolution: {actual_width}x{actual_height} @ {k4a_config.camera_fps.value}fps")
    else:
        print("Could not get color frame from Azure Kinect to determine resolution. Defaulting to 1280x720.")


except Exception as e:
    print(f"Error initializing Azure Kinect: {e}")
    # Ensure k4a is set to None if an error occurs during initialization
    k4a = None
    # No exit() here, allow the program to try and exit cleanly later

if k4a is None: # Exit if k4a initialization failed
    print("Failed to initialize Azure Kinect camera. Exiting.")
    exit()

window_name = 'Person Detection with Small Oval Thought Bubbles - Full Screen'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
print("Full screen mode activated! Press 'q' to quit, 'f' to toggle fullscreen.")

detector = PersonDetector()
frame_count = 0
last_time = time.time()
# Use the actual_width and actual_height obtained from the camera
scale_factor = min(actual_width / 640, actual_height / 480) * 0.7

while True:
    loop_start = time.time()

    # Get frame from Azure Kinect
    capture = k4a.get_capture()
    if capture.color is None:
        print("Error: Could not read color frame from Azure Kinect. Retrying...")
        time.sleep(0.01) # Small delay to avoid busy-waiting
        continue # Skip this frame

    # Convert BGRA to BGR if needed by pyk4a (Azure Kinect natively outputs BGRA)
    frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)

    frame_count += 1

    if frame_count % 1 == 0: # Process every frame
        if not detector.processing:
            detection_thread = threading.Thread(
                target=detector.detect_people,
                args=(frame.copy(),), # Pass a copy of the frame for thread safety
                daemon=True
            )
            detection_thread.start()

    with detector.detection_lock:
        current_people = detector.people.copy()

    for person in current_people:
        # Predict position for drawing even if not detected in the current frame
        box_to_draw = person.predict_position() if person.missed_frames > 0 else person.box
        x, y, w, h = box_to_draw

        # Ensure coordinates are within frame bounds before drawing
        x = max(0, x)
        y = max(0, y)
        w = min(frame.shape[1] - x, w)
        h = min(frame.shape[0] - y, h)

        if w > 0 and h > 0: # Only draw if box is valid
            bubble_x = x + w // 2
            bubble_y = y
            draw_thought_bubble(frame, bubble_x, bubble_y, person.thought,
                              person.bubble_animation, scale_factor)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        current_mode = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
        if current_mode == cv2.WINDOW_FULLSCREEN:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

if k4a:
    k4a.stop()
cv2.destroyAllWindows()
print("Camera released and windows closed.")