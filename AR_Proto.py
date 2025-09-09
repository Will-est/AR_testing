import cv2
import numpy as np
import time
import math

# Class to handle AR functionality
class ARScene:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Create AR marker detector (ArUco markers help detect flat surfaces)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Camera calibration parameters (ideally these should be calibrated for your specific camera)
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # 3D model data (simple cube vertices)
        self.object_points = np.array([
            [-0.1, -0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0],
            [-0.1, -0.1, 0.2], [0.1, -0.1, 0.2], [0.1, 0.1, 0.2], [-0.1, 0.1, 0.2]
        ], dtype=np.float32)
        
        # Cube edges for drawing
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Object placement state
        self.object_placed = False
        self.object_rvec = None
        self.object_tvec = None
        
        print("AR Scene initialized. Looking for flat surface (ArUco marker)...")

    def detect_marker(self, frame):
        """Detect ArUco marker to place object"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None:
            # Estimate pose of first marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 0.05, self.camera_matrix, self.dist_coeffs
            )
            
            if not self.object_placed:
                # Store the object position relative to marker
                self.object_rvec = rvecs[0]
                self.object_tvec = tvecs[0]
                self.object_placed = True
                print("Object placed on detected surface!")
            
            # Draw the detected marker
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            return True, frame
        
        return False, frame

    def render_object(self, frame):
        """Render the 3D object if it's been placed"""
        if not self.object_placed:
            return frame
        
        # Project 3D points to the image plane
        image_points, _ = cv2.projectPoints(
            self.object_points, self.object_rvec, self.object_tvec, 
            self.camera_matrix, self.dist_coeffs
        )
        
        # Draw the edges of the 3D object
        for edge in self.edges:
            pt1 = tuple(map(int, image_points[edge[0]].flatten()))
            pt2 = tuple(map(int, image_points[edge[1]].flatten()))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Add label
        cv2.putText(frame, "AR Object", tuple(map(int, image_points[0].flatten())), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame

    def run(self):
        """Main AR loop"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Try to detect markers if object isn't placed yet
            if not self.object_placed:
                detected, frame = self.detect_marker(frame)
            else:
                # If object is placed, continue tracking and rendering
                detected, frame = self.detect_marker(frame)
            
            # Render the AR object
            frame = self.render_object(frame)
            
            # Display status message
            status = "Object placed - Look around!" if self.object_placed else "Looking for flat surface..."
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show the frame
            cv2.imshow('AR Scene', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

# Run the AR application
if __name__ == "__main__":
    ar = ARScene()
    ar.run()