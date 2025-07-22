import cv2
import numpy as np
import threading
import time
import random
import math
from collections import deque

# Load YOLO model
try:
    net = cv2.dnn.readNet("./yolo_model/yolov3.weights", "./yolo_model/yolov3.cfg")
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Load bubble background image
try:
    bubble_bg = cv2.imread("bg.png", cv2.IMREAD_UNCHANGED)
    if bubble_bg is None:
        print("Warning: bg.png not found, using default bubble design")
        bubble_bg = None
    else:
        print("Bubble background loaded successfully!")
except Exception as e:
    print(f"Error loading bubble background: {e}")
    bubble_bg = None

# Random thoughts collection
RANDOM_THOUGHTS = [
    "I wonder what's for lunch today?",
    "Did I lock the door?",
    "This weather is nice!",
    "I need to call mom later",
    "Coffee sounds good right now",
    "What should I watch tonight?",
    "I should exercise more",
    "Time flies so fast!",
    "I love this song!",
    "Need to buy groceries",
    "Weekend plans?",
    "This is interesting!",
    "I'm feeling creative today",
    "Pizza or pasta tonight?",
    "Should I learn something new?",
    "Life is beautiful",
    "I need a vacation",
    "Technology is amazing!",
    "What a busy day!",
    "I'm grateful for today",
    "Maybe I'll try something new",
    "This looks fun!",
    "I wonder what time it is",
    "Should I get a snack?",
    "I love weekends!"
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
        self.prediction_box = box  # For smoother tracking
        self.velocity = (0, 0)  # Track movement for prediction
        self.missed_frames = 0
        
    def update(self, box, confidence):
        # Calculate velocity for prediction
        old_center = (self.box[0] + self.box[2]//2, self.box[1] + self.box[3]//2)
        new_center = (box[0] + box[2]//2, box[1] + box[3]//2)
        self.velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
        
        # Smooth box transition for less jitter
        alpha = 0.7  # Smoothing factor
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
        self.missed_frames = 0
        
        # Change thought every 4 seconds for more dynamic content
        if self.thought_timer > 120:  # Reduced from 150
            self.thought = random.choice(RANDOM_THOUGHTS)
            self.thought_timer = 0
    
    def predict_position(self):
        """Predict position when not detected for smoother tracking"""
        if self.missed_frames > 0:
            # Apply velocity-based prediction
            predicted_x = self.box[0] + self.velocity[0] * self.missed_frames
            predicted_y = self.box[1] + self.velocity[1] * self.missed_frames
            
            # Dampen velocity over time
            damping = 0.9 ** self.missed_frames
            self.velocity = (self.velocity[0] * damping, self.velocity[1] * damping)
            
            return [int(predicted_x), int(predicted_y), self.box[2], self.box[3]]
        return self.box

def calculate_overlap(box1, box2):
    """Calculate IoU overlap between two bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
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

def overlay_image_with_alpha(background, overlay, x, y):
    """Overlay an image with alpha channel onto background"""
    if overlay.shape[2] == 4:  # If overlay has alpha channel
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] = \
                (1 - alpha) * background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] + \
                alpha * overlay[:, :, c]
    else:
        background[y:y+overlay.shape[0], x:x+overlay.shape[1]] = overlay
    return background

def draw_thought_bubble(frame, x, y, text, animation_offset=0, scale_factor=1.0):
    """Draw a thought bubble with custom background and white text"""
    # Text properties - scaled for full screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * scale_factor
    font_thickness = max(1, int(2 * scale_factor))
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Bubble dimensions with padding - scaled
    padding = int(20 * scale_factor)
    bubble_width = text_width + padding * 2
    bubble_height = text_height + padding * 2
    
    # Animation effect
    animation_y = int(8 * math.sin(animation_offset) * scale_factor)
    bubble_x = x - bubble_width // 2
    bubble_y = y - bubble_height - int(30 * scale_factor) + animation_y
    
    # Ensure bubble stays within frame bounds
    bubble_x = max(10, min(bubble_x, frame.shape[1] - bubble_width - 10))
    bubble_y = max(10, bubble_y)
    
    # Draw bubble with custom background or fallback to default
    if bubble_bg is not None:
        # Resize bubble background to match bubble size
        resized_bg = cv2.resize(bubble_bg, (bubble_width, bubble_height))
        
        # Ensure we don't exceed frame boundaries
        end_x = min(bubble_x + bubble_width, frame.shape[1])
        end_y = min(bubble_y + bubble_height, frame.shape[0])
        actual_width = end_x - bubble_x
        actual_height = end_y - bubble_y
        
        if actual_width > 0 and actual_height > 0:
            # Crop the resized background if needed
            if actual_width < bubble_width or actual_height < bubble_height:
                resized_bg = resized_bg[:actual_height, :actual_width]
            
            # Apply the background
            if resized_bg.shape[2] == 4:  # Has alpha channel
                overlay_image_with_alpha(frame, resized_bg, bubble_x, bubble_y)
            else:
                frame[bubble_y:bubble_y+actual_height, bubble_x:bubble_x+actual_width] = resized_bg
    else:
        # Fallback to original bubble design
        # Draw bubble shadow
        shadow_offset = int(4 * scale_factor)
        cv2.ellipse(frame, 
                    (bubble_x + bubble_width//2 + shadow_offset, bubble_y + bubble_height//2 + shadow_offset),
                    (bubble_width//2, bubble_height//2), 0, 0, 360,
                    (50, 50, 50), -1)
        
        # Draw main bubble
        cv2.ellipse(frame, 
                    (bubble_x + bubble_width//2, bubble_y + bubble_height//2),
                    (bubble_width//2, bubble_height//2), 0, 0, 360,
                    (255, 255, 255), -1)
        
        # Draw bubble border
        cv2.ellipse(frame, 
                    (bubble_x + bubble_width//2, bubble_y + bubble_height//2),
                    (bubble_width//2, bubble_height//2), 0, 0, 360,
                    (180, 180, 180), max(2, int(3 * scale_factor)))
    
    # Draw thought bubble tail - scaled
    tail_x = x
    tail_y = y - int(15 * scale_factor)
    
    base_radii = [12, 8, 4]
    for i, radius in enumerate(base_radii):
        scaled_radius = int(radius * scale_factor)
        circle_y = tail_y - int(i * 12 * scale_factor) + animation_y // 2
        
        if bubble_bg is not None:
            # Use a small portion of the background for tail circles
            if bubble_bg.shape[2] == 4:  # Has alpha channel
                # Create small circular mask from background
                tail_size = scaled_radius * 2
                if tail_size <= bubble_bg.shape[0] and tail_size <= bubble_bg.shape[1]:
                    small_bg = cv2.resize(bubble_bg[:tail_size, :tail_size], (tail_size, tail_size))
                    mask = np.zeros((tail_size, tail_size), dtype=np.uint8)
                    cv2.circle(mask, (scaled_radius, scaled_radius), scaled_radius, 255, -1)
                    
                    # Apply circular background
                    for c in range(3):
                        roi = frame[circle_y-scaled_radius:circle_y+scaled_radius, 
                                   tail_x-scaled_radius:tail_x+scaled_radius, c]
                        if roi.shape == (tail_size, tail_size):
                            alpha_mask = (mask / 255.0) * (small_bg[:, :, 3] / 255.0)
                            roi[:] = (1 - alpha_mask) * roi + alpha_mask * small_bg[:, :, c]
            else:
                cv2.circle(frame, (tail_x, circle_y), scaled_radius, (200, 200, 200), -1)
        else:
            # Fallback tail design
            shadow_offset = max(1, shadow_offset//2)
            # Shadow
            cv2.circle(frame, (tail_x + shadow_offset, circle_y + shadow_offset), 
                      scaled_radius, (50, 50, 50), -1)
            # Main circle
            cv2.circle(frame, (tail_x, circle_y), scaled_radius, (255, 255, 255), -1)
            cv2.circle(frame, (tail_x, circle_y), scaled_radius, (180, 180, 180), 
                      max(1, int(2 * scale_factor)))
    
    # Draw text in WHITE color
    text_x = bubble_x + padding
    text_y = bubble_y + padding + text_height
    
    # Add text shadow for better readability
    shadow_offset = max(1, int(2 * scale_factor))
    cv2.putText(frame, text, (text_x + shadow_offset, text_y + shadow_offset), 
                font, font_scale, (0, 0, 0), font_thickness + 1)  # Black shadow
    
    # Main white text
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
                (255, 255, 255), font_thickness)  # White text

class PersonDetector:
    def __init__(self):
        self.net = net
        self.detections = []
        self.processing = False
        self.people = []
        self.next_person_id = 0
        self.detection_lock = threading.Lock()
        
    def detect_people(self, frame):
        """Process frame for person detection - optimized"""
        if self.processing:
            return
            
        self.processing = True
        
        try:
            # Get frame dimensions
            (height, width) = frame.shape[:2]
            
            # Create blob - optimized size for speed vs accuracy balance
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            
            # Perform forward propagation
            output_layer_names = self.net.getUnconnectedOutLayersNames()
            output_layers = self.net.forward(output_layer_names)
            
            # Initialize lists for detected people
            boxes = []
            confidences = []
            
            # Process detections
            for output in output_layers:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Class ID 0 = person, lowered threshold for better detection
                    if class_id == 0 and confidence > 0.35:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
            
            # Apply Non-Maximum Suppression
            current_detections = []
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.35, 0.25)
                if len(indices) > 0:
                    current_detections = [(boxes[i], confidences[i]) for i in indices.flatten()]
            
            # Update people tracking with thread safety
            with self.detection_lock:
                self.update_people_tracking(current_detections)
        
        except Exception as e:
            print(f"Detection error: {e}")
        finally:
            self.processing = False
        
    def update_people_tracking(self, detections):
        """Enhanced people tracking with prediction"""
        current_time = time.time()
        
        # Increment missed frames for all people
        for person in self.people:
            person.missed_frames += 1
        
        # Match detections with existing people
        matched_people = set()
        
        for box, confidence in detections:
            best_overlap = 0
            best_person = None
            
            # Find best matching existing person
            for i, person in enumerate(self.people):
                if i in matched_people:
                    continue
                    
                # Use predicted position if person was missed
                compare_box = person.predict_position() if person.missed_frames > 0 else person.box
                overlap = calculate_overlap(box, compare_box)
                
                if overlap > 0.25 and overlap > best_overlap:  # More forgiving threshold
                    best_overlap = overlap
                    best_person = person
            
            if best_person and best_overlap > 0.25:
                # Update existing person
                best_person.update(box, confidence)
                matched_people.add(self.people.index(best_person))
            else:
                # Create new person
                new_person = Person(self.next_person_id, box, confidence)
                self.people.append(new_person)
                self.next_person_id += 1
        
        # Remove people who haven't been seen for too long (increased timeout)
        self.people = [p for p in self.people if current_time - p.last_seen < 3.0]

# Initialize camera with maximum resolution and performance settings
cap = cv2.VideoCapture(0)

# Get maximum resolution supported by camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Try for Full HD
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)  # Try for higher FPS
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Better codec

# Get actual resolution achieved
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Camera resolution: {actual_width}x{actual_height} @ {actual_fps}fps")

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create full screen window
window_name = 'Person Detection with Thought Bubbles - Full Screen'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Full screen mode activated! Press 'q' to quit, 'f' to toggle fullscreen.")

# Initialize detector
detector = PersonDetector()

# Performance optimization variables
frame_count = 0
detection_interval = 2  # Process every 2nd frame for higher FPS
fps_counter = deque(maxlen=60)  # Larger sample for stable FPS
last_time = time.time()

# Calculate scale factor based on resolution
scale_factor = min(actual_width / 640, actual_height / 480)

while True:
    loop_start = time.time()
    
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break
    
    frame_count += 1
    
    # Run detection less frequently for higher display FPS
    if frame_count % detection_interval == 0:
        if not detector.processing:
            # Use threading for non-blocking detection
            detection_thread = threading.Thread(
                target=detector.detect_people, 
                args=(frame.copy(),), 
                daemon=True
            )
            detection_thread.start()
    
    # Thread-safe access to people data
    with detector.detection_lock:
        current_people = detector.people.copy()
    
    # Draw bounding boxes and thought bubbles
    person_count = len(current_people)
    
    for person in current_people:
        # Use predicted position for smoother tracking
        display_box = person.predict_position() if person.missed_frames > 0 else person.box
        x, y, w, h = display_box
        
        # Draw bounding box with thicker lines for full screen
        box_thickness = max(2, int(4 * scale_factor))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), box_thickness)
        
        # Add person label with larger font
        label = f"Person {person.id}: {person.confidence:.2f}"
        label_font_scale = 0.8 * scale_factor
        label_thickness = max(2, int(3 * scale_factor))
        cv2.putText(frame, label, (x, y - int(10 * scale_factor)), 
                   cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 255, 0), label_thickness)
        
        # Draw thought bubble
        bubble_x = x + w // 2
        bubble_y = y
        draw_thought_bubble(frame, bubble_x, bubble_y, person.thought, 
                          person.bubble_animation, scale_factor)
    
    # Calculate FPS
    current_time = time.time()
    if current_time > last_time:
        fps = 1.0 / (current_time - last_time)
        fps_counter.append(fps)
    last_time = current_time
    
    avg_fps = sum(fps_counter) / len(fps_counter) if fps_counter else 0
    
    # Display info with larger text for full screen
    info_font_scale = 1.2 * scale_factor
    info_thickness = max(2, int(3 * scale_factor))
    
    cv2.putText(frame, f"People: {person_count}", (20, int(50 * scale_factor)), 
                cv2.FONT_HERSHEY_SIMPLEX, info_font_scale, (0, 255, 255), info_thickness)
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, int(100 * scale_factor)), 
                cv2.FONT_HERSHEY_SIMPLEX, info_font_scale, (255, 255, 0), info_thickness)
    
    # Display the frame
    cv2.imshow(window_name, frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        # Toggle fullscreen
        current_mode = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
        if current_mode == cv2.WINDOW_FULLSCREEN:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed.")