import cv2
import numpy as np

class Plotter:
    def __init__(self) -> None:
        ...

    def draw_matches(self, image0: np.ndarray, image1: np.ndarray, kp1: list, kp2: list, 
                     matches: list, plot_title: str = None) -> None:
        result_image = cv2.drawMatches(image0, None, image1, None, matches, None)

        cv2.imshow('Matches', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_keypoints(self, image: np.ndarray, keypoints: list, plot_title: str = None) -> None:
        image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv2.imshow('KeyPoints', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return image
    