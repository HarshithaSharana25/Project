import mediapipe as mp 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import os

class PostureDetector:
    def __init__(self, angle_threshold=15):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.angle_threshold = angle_threshold
        self.angles = []
        self.bad_posture_counts = []

    # ... (keep your existing methods for calculate_pivot_angle and analyze_posture)

    def save_plots(self, save_dir):
        """Save the analysis plots to files."""
        if not self.angles:  # If no data collected, return
            return

        # Create plots directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Plot and save torso pivot angle
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.angles, label='Torso Pivot Angle', color='blue')
        plt.axhline(y=self.angle_threshold, color='red', linestyle='--', label='Threshold')
        plt.title('Torso Pivot Angle Over Time')
        plt.xlabel('Frames')
        plt.ylabel('Angle (degrees)')
        plt.legend()

        # Plot bad posture occurrences
        plt.subplot(1, 2, 2)
        plt.bar(range(len(self.bad_posture_counts)), self.bad_posture_counts, color='red')
        plt.title('Bad Posture Occurrences Over Time')
        plt.xlabel('Frames')
        plt.ylabel('Bad Posture Detected (1 = Yes, 0 = No)')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'posture_analysis.png'))
        plt.close()

        # Calculate and save summary statistics
        total_frames = len(self.angles)
        bad_posture_frames = sum(self.bad_posture_counts)
        bad_posture_percentage = (bad_posture_frames / total_frames * 100) if total_frames > 0 else 0
        average_angle = sum(self.angles) / len(self.angles) if self.angles else 0

        return {
            'total_frames': total_frames,
            'bad_posture_frames': bad_posture_frames,
            'bad_posture_percentage': bad_posture_percentage,
            'average_angle': average_angle
        }
    def calculate_pivot_angle(self, point_a, point_b):
        """Calculate the angle between the line formed by point_a and point_b with respect to the y-axis."""
        vector = np.array([point_a[0] - point_b[0], point_a[1] - point_b[1]])
        y_axis = np.array([0, 1])
        dot_product = np.dot(vector, y_axis)
        norm_vector = np.linalg.norm(vector)
        norm_y_axis = np.linalg.norm(y_axis)
        if norm_vector == 0 or norm_y_axis == 0:
            return 0.0
        cos_angle = dot_product / (norm_vector * norm_y_axis)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle

    def analyze_posture(self, frame):
        """Analyze the posture in the given frame and return the annotated image."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]

            if left_shoulder.visibility > 0.5 and left_hip.visibility > 0.5:
                image_height, image_width, _ = image.shape
                left_shoulder = (int(left_shoulder.x * image_width), int(left_shoulder.y * image_height))
                left_hip = (int(left_hip.x * image_width), int(left_hip.y * image_height))

                torso_angle = self.calculate_pivot_angle(left_hip, left_shoulder)
                self.angles.append(torso_angle)
                bad_posture = torso_angle > self.angle_threshold
                self.bad_posture_counts.append(1 if bad_posture else 0)

                cv2.putText(image, f"Torso Pivot Angle: {torso_angle:.2f}°",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if bad_posture:
                    cv2.putText(image, "Bad Posture Detected!", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    cv2.rectangle(image, (0, 0), (image_width, image_height), (0, 0, 255), 5)

        return image

    def plot_results(self):
        """Plot the results of the posture analysis."""
        plt.figure(figsize=(12, 5))

        # Plotting the torso pivot angle over time
        plt.subplot(1, 2, 1)
        plt.plot(self.angles, label='Torso Pivot Angle', color='blue')
        plt.axhline(y=self.angle_threshold, color='red', linestyle='--', label='Threshold')
        plt.title('Torso Pivot Angle Over Time')
        plt.xlabel('Frames')
        plt.ylabel('Angle (degrees)')
        plt.legend()

        # Plotting the bad posture occurrences
        plt.subplot(1, 2, 2)
        plt.bar(range(len(self.bad_posture_counts)), self.bad_posture_counts, color='red')
        plt.title('Bad Posture Occurrences Over Time')
        plt.xlabel('Frames')
        plt.ylabel('Bad Posture Detected (1 = Yes, 0 = No)')

        plt.tight_layout()
        plt.show()

def main():
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = PostureDetector(angle_threshold=10)  # Set your angle threshold here

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        analyzed_image = detector.analyze_posture(frame)
        cv2.imshow('Posture Detection', analyzed_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # After exiting the video stream, plot results
    detector.plot_results()

if __name__ == "__main__":
    main()
