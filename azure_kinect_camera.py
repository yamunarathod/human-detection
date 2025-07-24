import pyk4a
from pyk4a import PyK4A, Config
import cv2
import time

class AzureKinectCamera:
    """
    A class to manage the Azure Kinect camera.
    """
    def __init__(self):
        self.k4a = None
        self.width = 0
        self.height = 0
        self.is_initialized = False

    def start(self):
        """
        Initializes and starts the Azure Kinect camera.
        Returns True if successful, False otherwise.
        """
        try:
            # Define the configuration for Azure Kinect
            k4a_config = Config(
                color_resolution=pyk4a.ColorResolution.RES_720P, # 720p resolution
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED, # Narrow FOV unbinned depth
                camera_fps=pyk4a.FPS.FPS_30 # 30 frames per second
            )
            self.k4a = PyK4A(k4a_config) # Create PyK4A object with the defined config
            self.k4a.start() # Start the camera
            print("Azure Kinect camera opened successfully!")

            # Get actual resolution from the first capture
            capture = self.k4a.get_capture()
            if capture.color is not None:
                self.height, self.width, _ = capture.color.shape
                print(f"Azure Kinect resolution: {self.width}x{self.height} @ {k4a_config.camera_fps.value}fps")
                self.is_initialized = True
            else:
                print("Could not get color frame from Azure Kinect to determine resolution.")
                self.k4a.stop() # Stop camera if no frame is available
                self.k4a = None # Reset k4a object
                self.is_initialized = False

        except Exception as e:
            print(f"Error initializing Azure Kinect: {e}")
            if self.k4a:
                self.k4a.stop()
            self.k4a = None
            self.is_initialized = False
        
        return self.is_initialized

    def get_frame(self):
        """
        Captures a frame from the Azure Kinect camera.
        Returns a BGR image frame or None if an error occurs.
        """
        if not self.is_initialized or self.k4a is None:
            return None

        try:
            capture = self.k4a.get_capture()
            if capture.color is not None:
                # Azure Kinect natively outputs BGRA, convert to BGR for OpenCV
                frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)
                return frame
            else:
                print("Error: Could not read color frame from Azure Kinect.")
                return None
        except Exception as e:
            print(f"Error getting frame from Azure Kinect: {e}")
            return None

    def stop(self):
        """
        Stops the Azure Kinect camera and releases resources.
        """
        if self.k4a:
            self.k4a.stop()
            print("Azure Kinect camera released.")
        self.is_initialized = False
        self.k4a = None

    def get_resolution(self):
        """
        Returns the resolution (width, height) of the camera.
        """
        return self.width, self.height

# Example usage (for testing this module directly)
if __name__ == "__main__":
    camera = AzureKinectCamera()
    if camera.start():
        try:
            while True:
                frame = camera.get_frame()
                if frame is not None:
                    cv2.imshow("Azure Kinect Feed", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        finally:
            camera.stop()
            cv2.destroyAllWindows()
    else:
        print("Failed to start Azure Kinect camera.")

