import cv2
import numpy as np


def read_and_convert_image(image_path):
    """
    Read an image from the specified path and convert it to RGB format.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        np.ndarray: The image in RGB format.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"Error reading image from {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"Error in read_and_convert_image: {e}")
        return None


def detect_keypoints_and_descriptors(image, detector):
    """
    Detect keypoints and compute descriptors for the given image using a specified detector.

    Parameters:
        image (np.ndarray): The input image.
        detector: The keypoint detector (e.g., cv2.SIFT_create()).

    Returns:
        tuple: Keypoints and descriptors.
    """
    try:
        # Detect keypoints and compute descriptors
        keypoints, descriptors = detector.detectAndCompute(image, None)
        return keypoints, descriptors

    except Exception as e:
        print(f"Error in detect_keypoints_and_descriptors: {e}")
        return None, None


def match_keypoints(descriptor_matcher, descriptors_obj, descriptors_scene):
    """
    Match keypoints between two sets of descriptors using a specified descriptor matcher.

    Parameters:
        descriptor_matcher: The descriptor matcher (e.g., cv2.DescriptorMatcher_create()).
        descriptors_obj: Descriptors of the target object.
        descriptors_scene: Descriptors of the scene.

    Returns:
        list: List of good matches.
    """
    # Perform k-nearest neighbors matching
    knn_matches = descriptor_matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
    # Apply ratio test to obtain good matches
    ratio_thresh = 0.75
    good_matches = [m for m, n in knn_matches if m.distance < ratio_thresh * n.distance]
    return good_matches


def localize_object(keypoints_obj, keypoints_scene, good_matches, target_shape):
    """
    Localize the object in the scene using homography estimation.

    Parameters:
        keypoints_obj: Keypoints of the target object.
        keypoints_scene: Keypoints of the scene.
        good_matches: List of good matches.
        target_shape: Shape of the target object.

    Returns:
        np.ndarray: Homography matrix.
    """
    # Extract matched keypoints
    obj = np.float32([keypoints_obj[m.queryIdx].pt for m in good_matches])
    scene = np.float32([keypoints_scene[m.trainIdx].pt for m in good_matches])

    # Find homography matrix using RANSAC
    H, _ = cv2.findHomography(obj, scene, cv2.RANSAC)

    # Define object corners
    obj_corners = np.float32(
        [
            [0, 0],
            [target_shape[1], 0],
            [target_shape[1], target_shape[0]],
            [0, target_shape[0]],
        ]
    )

    # Transform object corners to scene coordinates
    scene_corners = cv2.perspectiveTransform(obj_corners.reshape(-1, 1, 2), H)

    return scene_corners
