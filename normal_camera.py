import cv2
import time

class NormalCamera:
    """
    A class to manage a standard webcam (using cv2.VideoCapture).
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.width = 0
        self.height = 0
        self.is_initialized = False

    def start(self):
        """
        Initializes and starts the standard webcam.
        Returns True if successful, False otherwise.
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open webcam with index {self.camera_index}")

            # Set resolution (optional, may not be supported by all cameras)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.is_initialized = True
            print(f"Normal camera (index {self.camera_index}) opened successfully! Resolution: {self.width}x{self.height}")
            return True
        except Exception as e:
            print(f"Error initializing normal camera: {e}")
            if self.cap:
                self.cap.release()
            self.is_initialized = False
            return False

    def get_frame(self):
        """
        Captures a frame from the standard webcam.
        Returns a BGR image frame or None if an error occurs.
        """
        if not self.is_initialized or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame from normal camera.")
            return None
        return frame

    def stop(self):
        """
        Stops the standard webcam and releases resources.
        """
        if self.cap:
            self.cap.release()
            print("Normal camera released.")
        self.is_initialized = False
        self.cap = None

    def get_resolution(self):
        """
        Returns the resolution (width, height) of the camera.
        """
        return self.width, self.height

# Example usage (for testing this module directly)
if __name__ == "__main__":
    camera = NormalCamera(camera_index=0) # Use camera index 0 (default webcam)
    if camera.start():
        try:
            while True:
                frame = camera.get_frame()
                if frame is not None:
                    cv2.imshow("Normal Camera Feed", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        finally:
            camera.stop()
            cv2.destroyAllWindows()
    else:
        print("Failed to start normal camera.")

