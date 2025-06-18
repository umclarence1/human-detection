import cv2
import mediapipe as mp
import time
import os
import platform
import threading
from datetime import datetime

class HumanDetector:
    def __init__(self, duration=15, confidence_threshold=0.5):
        """
        Initialize the Human Detection System
        
        Args:
            duration (int): Detection duration in seconds
            confidence_threshold (float): Minimum confidence for detection (0.0-1.0)
        """
        self.duration = duration
        self.confidence_threshold = confidence_threshold
        
        # Setup MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Detection stats
        self.detection_count = 0
        self.total_frames = 0
        self.last_detection_time = None
        
        # Sound system setup
        self.setup_sound_system()
        
    def setup_sound_system(self):
        """Setup cross-platform sound system"""
        self.sound_available = False
        system = platform.system().lower()
        
        if system == "windows":
            try:
                import winsound
                self.winsound = winsound
                self.sound_available = True
                self.sound_method = "windows"
            except ImportError:
                print("⚠️  winsound not available")
        elif system in ["linux", "darwin"]:  # Linux or macOS
            if os.system("which aplay > /dev/null 2>&1") == 0:  # Linux
                self.sound_method = "linux"
                self.sound_available = True
            elif os.system("which afplay > /dev/null 2>&1") == 0:  # macOS
                self.sound_method = "macos"
                self.sound_available = True
        
        if not self.sound_available:
            print("⚠️  Sound alerts not available on this system")
    
    def play_sound(self):
        """Play detection sound based on platform"""
        if not self.sound_available:
            return
            
        def sound_thread():
            try:
                if self.sound_method == "windows":
                    self.winsound.Beep(1000, 200)
                elif self.sound_method == "linux":
                    os.system("aplay -q /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null &")
                elif self.sound_method == "macos":
                    os.system("afplay /System/Library/Sounds/Glass.aiff &")
            except Exception as e:
                print(f"⚠️  Sound error: {e}")
        
        # Play sound in separate thread to avoid blocking
        threading.Thread(target=sound_thread, daemon=True).start()
    
    def log_detection(self, detected):
        """Log detection with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if detected:
            self.detection_count += 1
            self.last_detection_time = time.time()
            print(f"[{timestamp}] ✅ Human detected (Total: {self.detection_count})")
        
        self.total_frames += 1
    
    def draw_info_panel(self, img, detected, elapsed_time, fps):
        """Draw information panel on the image"""
        h, w, _ = img.shape
        
        # Background panel - larger and more prominent
        panel_width = 500
        panel_height = 160
        cv2.rectangle(img, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (panel_width, panel_height), (0, 255, 255), 3)  # Yellow border
        
        # Detection status - larger text
        status_text = "✅ HUMAN DETECTED" if detected else "❌ NO HUMAN DETECTED"
        status_color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(img, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        
        # Time remaining with progress bar
        time_left = max(0, self.duration - elapsed_time)
        time_text = f"Time Left: {time_left:.1f}s / {self.duration}s"
        cv2.putText(img, time_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Progress bar
        progress = min(1.0, elapsed_time / self.duration)
        bar_width = 300
        bar_height = 8
        bar_x, bar_y = 20, 85
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 255), -1)
        
        # Statistics
        detection_rate = (self.detection_count / self.total_frames * 100) if self.total_frames > 0 else 0
        cv2.putText(img, f"Detections: {self.detection_count} ({detection_rate:.1f}%)", 
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"FPS: {fps:.1f} | Frames: {self.total_frames}", 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Last detection time
        if self.last_detection_time:
            time_since = time.time() - self.last_detection_time
            cv2.putText(img, f"Last Detection: {time_since:.1f}s ago", 
                       (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Controls info in bottom right
        controls = ["Q: Quit", "S: Screenshot", "R: Reset"]
        for i, control in enumerate(controls):
            cv2.putText(img, control, (w - 200, h - 60 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def run(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam.")
            return False
        
        # Set camera properties for better performance and larger window
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Webcam opened successfully. Starting detection...")
        print(f"⏱️  Detection will run for {self.duration} seconds")
        print("🎯 Press 'q' to quit early | 's' for screenshot | 'r' to reset timer")
        print("🖼️  Webcam window will open automatically...")
        
        # Create named window and make it resizable
        cv2.namedWindow("Human Detection System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Human Detection System", 1280, 720)
        
        start_time = time.time()
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Check if duration exceeded
                if elapsed_time >= self.duration:
                    print("⏰ Time limit reached. Exiting...")
                    break
                
                # Read frame
                success, img = cap.read()
                if not success:
                    print("❌ Failed to read from webcam.")
                    break
                
                # Flip image horizontally for mirror effect
                img = cv2.flip(img, 1)
                
                # Convert to RGB for MediaPipe
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.pose.process(img_rgb)
                
                # Check for human detection
                detected = False
                if results.pose_landmarks:
                    # Calculate confidence based on visible landmarks
                    visible_landmarks = sum(1 for lm in results.pose_landmarks.landmark if lm.visibility > 0.5)
                    if visible_landmarks >= 10:  # Require at least 10 visible landmarks
                        detected = True
                        
                        # Draw pose landmarks with enhanced visualization
                        # Draw connections (skeleton)
                        self.mp_draw.draw_landmarks(
                            img, 
                            results.pose_landmarks, 
                            self.mp_pose.POSE_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=1),  # Yellow connections
                            self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=4)   # Magenta joints
                        )
                        
                        # Draw bounding box around detected person
                        h, w, _ = img.shape
                        landmarks = results.pose_landmarks.landmark
                        
                        # Get bounding box coordinates
                        x_coords = [lm.x * w for lm in landmarks if lm.visibility > 0.5]
                        y_coords = [lm.y * h for lm in landmarks if lm.visibility > 0.5]
                        
                        if x_coords and y_coords:
                            x_min, x_max = int(min(x_coords)), int(max(x_coords))
                            y_min, y_max = int(min(y_coords)), int(max(y_coords))
                            
                            # Add padding to bounding box
                            padding = 20
                            x_min = max(0, x_min - padding)
                            y_min = max(0, y_min - padding)
                            x_max = min(w, x_max + padding)
                            y_max = min(h, y_max + padding)
                            
                            # Draw bounding box
                            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                            
                            # Draw confidence text on bounding box
                            confidence_text = f"Human: {visible_landmarks} points"
                            cv2.putText(img, confidence_text, (x_min, y_min - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Play sound alert
                        self.play_sound()
                
                # Log detection
                self.log_detection(detected)
                
                # Calculate FPS
                fps_frame_count += 1
                if current_time - fps_start_time >= 1.0:
                    current_fps = fps_frame_count / (current_time - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = current_time
                
                # Draw information panel
                self.draw_info_panel(img, detected, elapsed_time, current_fps)
                
                # Show image
                cv2.imshow("Human Detection System", img)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Exiting early...")
                    break
                elif key == ord('s'):
                    # Take screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, img)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('r'):
                    # Reset timer
                    start_time = time.time()
                    print("🔄 Timer reset!")
        
        except KeyboardInterrupt:
            print("\n⚠️  Detection interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.pose.close()
            
            # Print final statistics
            print("\n" + "="*50)
            print("📊 DETECTION SUMMARY")
            print("="*50)
            print(f"🕐 Total runtime: {elapsed_time:.1f} seconds")
            print(f"🎯 Total detections: {self.detection_count}")
            print(f"📊 Detection rate: {(self.detection_count/self.total_frames*100):.1f}%" if self.total_frames > 0 else "N/A")
            print(f"🖼️  Total frames processed: {self.total_frames}")
            print(f"⚡ Average FPS: {(self.total_frames/elapsed_time):.1f}" if elapsed_time > 0 else "N/A")
            print("="*50)


def main():
    """Main function to run the human detector"""
    print("🤖 Human Detection System v2.0")
    print("="*40)
    print("🎯 Starting with fixed settings:")
    print("   - Duration: 60 seconds")
    print("   - Confidence: 0.5")
    print("   - Auto-start enabled")
    print("="*40)
    
    # Fixed configuration - no user input required
    duration = 60  # Fixed 60 seconds
    confidence = 0.5  # Fixed confidence threshold
    
    # Create and run detector
    detector = HumanDetector(duration=duration, confidence_threshold=confidence)
    detector.run()


if __name__ == "__main__":
    main()