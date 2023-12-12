import argparse
import cv2
from utils.image_processing import (
    read_and_convert_image,
    detect_keypoints_and_descriptors,
    match_keypoints,
    localize_object,
)


def process_images(target_path, frame_path):
    target = read_and_convert_image(target_path)
    frame = read_and_convert_image(frame_path)

    detector = cv2.SIFT_create()

    keypoints_obj, descriptors_obj = detect_keypoints_and_descriptors(target, detector)
    keypoints_scene, descriptors_scene = detect_keypoints_and_descriptors(
        frame, detector
    )

    descriptor_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    good_matches = match_keypoints(
        descriptor_matcher, descriptors_obj, descriptors_scene
    )

    scene_corners = localize_object(
        keypoints_obj, keypoints_scene, good_matches, target.shape
    )

    center_point = scene_corners.mean(axis=(0, 1))
    return center_point, frame


def main():
    parser = argparse.ArgumentParser(description="Target object detection")
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        help="Path to the target image",
        required=False,
        default="images/target.jpeg",
    )
    parser.add_argument(
        "-f",
        "--frame",
        type=str,
        help="Path to the frame image",
        required=False,
        default="images/frame.jpeg",
    )
    parser.add_argument(
        "-o",
        "--output_image",
        type=str,
        help="Path to the resulting image",
        required=False,
        default="out.jpeg",
    )

    args = parser.parse_args()

    center_point, frame = process_images(args.target, args.frame)
    center_point = [int(i) for i in center_point]

    cv2.circle(frame, center_point, 10, (255, 0, 0), -1)
    cv2.imwrite(args.output_image, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
