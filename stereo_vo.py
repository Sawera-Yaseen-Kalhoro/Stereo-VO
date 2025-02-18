import cv2
import os
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader import KITTIDataset
from plotter import Plotter

from typing import Tuple, Dict, List
import matplotlib.pyplot as plt


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

class StereoOdometry:
    """
    Important Attributes:
        pose (ndarray): The current pose matrix (4x4) of the camera.
        projMatr1 (ndarray): The 3x4 projection matrix for the first camera.
        projMatr2 (ndarray): The 3x4 projection matrix for the second camera.
        camera_matrix (ndarray): The intrinsic camera matrix (3x3).

    Methods:
        calculate_camera_parameters(): Extracts camera parameters from projection matrices.
        load_camera_matrices(sequence): Loads camera projection matrices for a given sequence.
        triangulate_points(matches, keypoints0, keypoints1): Triangulates 3D points from matched keypoints in stereo images.
        compute_pnp(Xk_minus_1, keypoints1, matches): Computes camera pose using the Perspective-n-Point algorithm.
        filter_matches(matches, threshold=30): Filters matches based on a distance threshold.
        project_matches_to_3d(matches, keypoints0, keypoints1, camera_matrix): Projects matched keypoints into 3D space.
        construct_se3_matrix(rotation_vector, translation_vector): Constructs an SE(3) matrix from rotation and translation vectors.
        concatenate_transform(transform1, transform2): Concatenates two SE(3) transformation matrices.
        detect_features(image): Detects SIFT features in an image.
        find_matches(keypoints0, keypoints1, descriptors0, descriptors1): Finds feature matches using BFMatcher.
        save_pose(file, pose): Saves the camera pose in KITTI format to a file.
        run(p_results): Executes the stereo odometry algorithm on the dataset.
        main(): Entry point for running the class on a dataset sequence.
    """

    def __init__(self) -> None:
        # self.base_dir = r"C:\Users\Fatima Rustamani\Desktop\Stereo_VO"
        self.base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.base_dir = os.path.join(self.base_dir, "Stereo_VO")
        self.pose = np.eye(4, dtype=np.float64)
        self.results_dir = os.path.join(self.base_dir, "dataset", "results")
        self.colors = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255),}

        self.camera_matrix = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02],
                                       [0.000000e+00, 7.215377e+02, 1.728540e+02],
                                       [0.000000e+00, 0.000000e+00, 1.000000e+00]])
        
        self.config = self.load_config(os.path.join(self.base_dir, "config", "cfg.yaml"))
        
        self.threshold = self.config["filter_threshold"]
        self.plotter = Plotter()
        self.show_plots = self.config["show_plots"]
        self.min_matches = self.config["min_matches"]
        self.translation_history = []
        self.sequence = self.config["sequence"]
        self.dataset = KITTIDataset(self.sequence, self.base_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.projMatr1, self.projMatr2 = self.load_projection_matrices(self.sequence)
        self.intrinsic_params1 = self.calculate_camera_parameters(self.projMatr1)


    def calculate_camera_parameters(self, P0: np.ndarray):
        """
        Calculates the camera parameters from the projection matrices. 
        """
        # R & T matrices too
        intrinsic, rotation, translation, _, _, _, _ = cv2.decomposeProjectionMatrix(P0)
        return intrinsic
        
    def load_projection_matrices(self, sequence: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the projection matrices for a given sequence.
        """
        calib_file_path = os.path.join(self.base_dir, "dataset", "sequences", f"{sequence:02d}", "calib.txt")
        try:
            with open(calib_file_path, 'r') as file:
                lines = file.readlines()
                P0 = np.array([float(value) for value in lines[0].split()[1:13]]).reshape(3, 4)
                P1 = np.array([float(value) for value in lines[1].split()[1:13]]).reshape(3, 4)
        except Exception:
            print(f"Could not read {calib_file_path}")
        return P0, P1
    
    def load_config(self, config_file: str) -> dict:
        """
        Loads the configuration file (having system params).
        """
        try:
            with open(config_file, "r") as file:
                config = yaml.safe_load(file)
        except Exception:
            print(f"Could not read {config_file}")
        return config
    
    def plot3d_trajectory(self, translation_history: list, image_index: int) -> None:
        """
        Plots the 3D trajectory based on the translation history.
        """
        trajectory = np.array(translation_history)
        X = trajectory[:, 0]
        Y = trajectory[:, 1]
        Z = trajectory[:, 2]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X, Y, Z, color='b', label='Trajectory')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Trajectory')
        ax.legend()

        plt.show()

    def triangulate_points(self, matches: list, keypoints0: list, keypoints1: list) -> np.ndarray:
        """
        Triangulates 3D points from stereo image pairs using matched keypoints.

        Returns:
        - np.ndarray: A 2D array of shape (N, 3), where N is the number of matched keypoints. Each row in the array represents the 3D coordinates (X, Y, Z)
        of a point triangulated from the corresponding matched keypoints in the stereo image pair.The function first converts the matched keypoints into 2xN arrays of 2D points (N being
        the number of matches). The function then performs triangulation to get 3D points in a homogeneous coordinate system and converts them to a standard 3D coordinate system.
        """
        # need - proj matrix + KP + matches

        # Convert keypoints to the required format
        points0 = np.float32([keypoints0[m[0].queryIdx].pt for m in matches]).reshape(-1, 2)
        points1 = np.float32([keypoints1[m[0].trainIdx].pt for m in matches]).reshape(-1, 2)

        # Triangulate points
        points4D = cv2.triangulatePoints(self.projMatr1, self.projMatr2, points0.T, points1.T)   # 4D - X,Y,Z and scale
        points3D = cv2.convertPointsFromHomogeneous(points4D.T)
        return points3D.reshape(-1, 3)
    
    def compute_pnp(self, Xk_minus_1: np.ndarray, keypoints1: list, matches: list) -> np.ndarray:
        """
        Computes the Perspective-n-Point (PnP) algorithm to estimate the camera pose.

        Important Args:
            Xk_minus_1 (np.ndarray): The previous 3D points in the world coordinate system.

        Returns:
            np.ndarray: The estimated camera pose as a 4x4 transformation matrix.
        """
        image_points = np.float32([keypoints1[m[0].trainIdx].pt for m in matches])
        object_points = Xk_minus_1[:len(image_points)]
        # Object pose from 3D-2D point correspondences using RANSAC
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(object_points, image_points, self.intrinsic_params1[:, :3], None, flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.9999, reprojectionError=1)
        
        if success:
            # Pose refinement using Levenberg-Marquardt optimization
            rotation_vector, translation_vector = cv2.solvePnPRefineLM(
                object_points[inliers], image_points[inliers], 
                self.intrinsic_params1[:, :3], None, rotation_vector, translation_vector)

        return self.construct_se3_matrix(rotation_vector, translation_vector)
    

    def construct_se3_matrix(self, rotation_vector: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
        """
        SE(3) matrix from R & T vectors
        """
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        se3_matrix = np.eye(4)
        se3_matrix[:3, :3] = rotation_matrix
        se3_matrix[:3, 3] = translation_vector.flatten()
        return se3_matrix 

    def concatenate_transform(self, transform1: np.ndarray, transform2: np.ndarray) -> np.ndarray:
        """
        Concatenate two SE(3) transformations.
        """
        return np.dot(transform1, transform2)

    
    def detect_features(self, image: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Detects features in the given image using the SIFT algorithm.
        """
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors
    
    
    def find_matches(self, keypoints0: list, keypoints1: list, descriptors0: np.ndarray, descriptors1: np.ndarray) -> list:
        """
        Finds matches between keypoints and descriptors using the BFMatcher algorithm.
        """
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(descriptors0, descriptors1, k=2)
        return matches
    
    def filter_matches(self, matches: list, threshold: float=0.75) -> list:
        """
        Filter out matches that have a distance greater than a threshold (75%).
        """
        good_matches = []
        for m, n in matches:
            # Lowe's ratio test
            if m.distance < threshold * n.distance:
                good_matches.append([m])
        return good_matches

    
    def plot_title(self, name: str, sequence: int, idx: Dict[str, int]) -> str:
        """
        Creates a title for the image.
        """
        plots = {"Keypoints": f"{name} - Sequence {sequence:02d} - Image {idx}", 
                    "Matches": f"{name} - Sequence {sequence:02d} - Image {idx}",
                    "Trajectory": f"{name} - Sequence {sequence:02d}"
                }
        return plots[name.capitalize()]

    def save_pose(self, file: str, pose: np.ndarray) -> None:
        """
        Saves the pose in the KITTI format to the specified file - for later plotting and evaluating trajectories with EVO.
        """
        if file and pose.shape == (4, 4) and pose.dtype == np.float64:
            file.write(
                f"{pose[0,0]} {pose[0,1]} {pose[0,2]} {pose[0,3]} "
                f"{pose[1,0]} {pose[1,1]} {pose[1,2]} {pose[1,3]} "
                f"{pose[2,0]} {pose[2,1]} {pose[2,2]} {pose[2,3]}\n")


    def run(self, results_filepath: str) -> None:
        """
        Execution of the stereo odometry algorithm on a sequence of stereo images.

        Steps:
        1. Detect Features
        2. Find Matches
        3. Filter Matches
        4. Triangulate points
        5. Compute PnP
        """
        is_paused = False
        for image_index, (left_image, right_image) in enumerate(tqdm(self.dataloader)):
            # Convert the images to NumPy arrays
            left_image = left_image.numpy().squeeze().astype(np.uint8)
            right_image = right_image.numpy().squeeze().astype(np.uint8)

            if left_image is None or right_image is None:
                break

            # Save the first image to start feature matching with the second set
            if image_index == 0:
                self.save_pose(results_filepath, self.pose)
                previous_left_image = left_image
                previous_right_image = right_image
                continue

            # Control for pausing the process
            key_press = cv2.waitKey(1000000 if is_paused else 1)
            if key_press == ord(' '):
                is_paused = not is_paused

            # Detect features in the previous and current left images
            previous_left_keypoints, previous_left_descriptors = self.detect_features(previous_left_image)
            previous_right_keypoints, previous_right_descriptors = self.detect_features(previous_right_image)
            current_left_keypoints, current_left_descriptors = self.detect_features(left_image)

            # Draw keypoints
            if self.show_plots:
                self.plotter.draw_keypoints(previous_left_image, previous_left_keypoints, self.plot_title("Keypoints", self.sequence, {"left": f"{image_index-1:06d}"}))
                self.plotter.draw_keypoints(previous_right_image, previous_right_keypoints, self.plot_title("Keypoints", self.sequence, {"right": f"{image_index-1:06d}"}))

            # Find matches between previous and current left images
            left_matches = self.find_matches(previous_left_keypoints, current_left_keypoints, previous_left_descriptors, current_left_descriptors)
            filtered_left_matches = self.filter_matches(left_matches, self.threshold)

            # Draw matches
            if self.show_plots:
                self.plotter.draw_matches(previous_left_image, left_image, previous_left_keypoints, current_left_keypoints, filtered_left_matches, 
                                self.plot_title("Matches", self.sequence, {"left": f"{image_index-1:06d}", "right": f"{image_index:06d}"}))

            # Keep track of matched keypoints for L,k-1 and R,k-1
            previous_matched_left_points = [previous_left_keypoints[match[0].queryIdx].pt for match in filtered_left_matches]
            current_matched_left_points = [current_left_keypoints[match[0].trainIdx].pt for match in filtered_left_matches]

            # STEREO MATCHING - needed for depth - Find matches between previous left and right images
            left_to_right_matches = self.find_matches(previous_left_keypoints, previous_right_keypoints, previous_left_descriptors, previous_right_descriptors)
            filtered_left_to_right_matches = self.filter_matches(left_to_right_matches, self.threshold)

            # Keep track of matched keypoints between previous left and right images
            matched_left_points_in_left_to_right = [previous_left_keypoints[match[0].queryIdx].pt for match in filtered_left_to_right_matches]
            matched_right_points_in_left_to_right = [previous_right_keypoints[match[0].trainIdx].pt for match in filtered_left_to_right_matches]

            # Find indices of matched L,k-1 in step 2 in L,k-1 from step 1
            matched_indices = [(i, previous_matched_left_points.index(item)) for i, item in enumerate(matched_left_points_in_left_to_right) if item in previous_matched_left_points]
             
            filtered_current_left_matches = [filtered_left_matches[i[1]] for i in matched_indices]

            # Triangulate and compute the pose if sufficient matches are found
            if len(filtered_left_to_right_matches) > self.min_matches:
                triangulated_points = self.triangulate_points([filtered_left_to_right_matches[i[0]] for i in matched_indices], 
                                                            previous_left_keypoints, previous_right_keypoints)
                
                # Project points from step 5 to L,k in step 4
                Tk = self.compute_pnp(triangulated_points, current_left_keypoints, filtered_current_left_matches)

                # Invert the Tk matrix 
                Tk = np.linalg.inv(Tk)

                # Concatenate Tk to the previous pose
                self.pose = self.concatenate_transform(self.pose, Tk)
                
                self.save_pose(results_filepath, self.pose)

                # for plotting trajectory
                self.translation_history.append(self.pose[:3, 3].flatten())
            else:
                    print("Not enough matches to compute pose.")

            # Update previous images for next iteration
            previous_left_image = left_image
            previous_right_image = right_image
        
        # 3D Plot of the final trajectory
        self.plot3d_trajectory(self.translation_history, image_index)

    def main(self):
        create_dir(self.results_dir)
        with open(os.path.join(self.results_dir, f"{self.sequence:02d}.txt"), "w") as p_results:
            self.run(p_results)

        print(f"Results written to {os.path.join(self.results_dir, f'{self.sequence:02d}.txt')}")
        
if __name__ == "__main__":
    pose_estimator = StereoOdometry()
    pose_estimator.main()
