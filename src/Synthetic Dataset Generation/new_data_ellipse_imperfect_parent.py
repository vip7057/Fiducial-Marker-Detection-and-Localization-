##CONCENTRIC ELLIPSES WITH NOISE##
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# Function to create a binary ellipse edge image with noise, occlusions, and additional shapes
def create_imperfect_parent_ellipse(size, center, axes, angle):
    image = np.zeros(size, dtype=np.uint8)  # Create a black image

    t = np.random.randint(1,4)

    # Draw the imperfect parent ellipse
    cv2.ellipse(image, tuple(center), tuple(axes), angle, 0, 360, 255, t)

    # Introduce imperfections in the parent ellipse by adding noise
    noise = np.random.normal(0, 5, size=image.shape).astype(int)
    image = np.clip(image + noise, 0, 255)

    # Calculate child ellipse parameters
    child_axes_ratio = np.random.uniform(1.2, 1.6)  # Ratio between child and parent ellipse axes
    child_axes = (axes * (1/child_axes_ratio)).astype(int)
    child_angle = angle  # Make the child ellipse have the same orientation as the parent ellipse

    # # Draw one perfectly concentric child ellipse
    cv2.ellipse(image, tuple(center), tuple(child_axes), child_angle, 0, 360, 255, t)


    # # Add tiny random shapes as noise
    # for _ in range(np.random.randint(3, 6)):
    #     shape_center = np.random.randint(0, size, size=(2,))
    #     shape_size = np.random.randint(2, 6)
    #     cv2.rectangle(image, tuple(shape_center - shape_size),
    #                   tuple(shape_center + shape_size), 255, 1)
    #
    #     shape_type = np.random.choice(['circle', 'triangle', 'both'])
    #
    # if shape_type == 'circle':
    #     shape_center = np.random.randint(0, size, size=(2,))
    #     cv2.circle(image, tuple(shape_center), shape_size, 255, 1)
    # elif shape_type == 'triangle':
    #     shape_center = np.random.randint(0, size, size=(2,))
    #     triangle_pts = np.array([
    #         shape_center + [0, -shape_size],
    #         shape_center + [-shape_size, shape_size],
    #         shape_center + [shape_size, shape_size]
    #     ], dtype=np.int32)
    #     cv2.polylines(image, [triangle_pts], isClosed=True, color=255, thickness=1)
    # else:
    #     shape_center = np.random.randint(0, size, size=(2,))
    #     cv2.circle(image, tuple(shape_center), shape_size, 255, 1)
    #     shape_center = np.random.randint(0, size, size=(2,))
    #     triangle_pts = np.array([
    #         shape_center + [0, -shape_size],
    #         shape_center + [-shape_size, shape_size],
    #         shape_center + [shape_size, shape_size]
    #     ], dtype=np.int32)
    #     cv2.polylines(image, [triangle_pts], isClosed=True, color=255, thickness=1)
    #
    # # Add random lines
    # for _ in range(np.random.randint(2, 5)):
    #     line_start = np.random.randint(0, size, size=(2,))
    #     line_end = np.random.randint(0, size, size=(2,))
    #     cv2.line(image, tuple(line_start), tuple(line_end), 255, 1)

    # Optionally add salt noise
    if np.random.rand() < 0.3:
        salt_noise = np.random.rand(*size) < 0.05
        image[salt_noise] = 255  # Set pixels to white (255) where salt noise is present

    return image, child_axes_ratio  # Return child_axes_ratio


def save_image_with_labels(image, presence_label, outer_bbox_label, save_path_image, save_path_outer_label, save_path_ellipse_label):
    """
    Save the binary ellipse edge image to a file along with labels.

    Parameters:
    - image: Numpy array, binary ellipse edge image.
    - presence_label: String, label information for ellipse presence.
    - outer_bbox_label: String, label information for outer ellipse.
    - save_path_image: String, path to save the image.
    - save_path_outer_label: String, path to save the outer label file.
    - save_path_ellipse_label: String, path to save the ellipse label file.
    """

    # Save the image
    cv2.imwrite(save_path_image, image)

    # Save the label information for outer ellipse
    with open(save_path_outer_label, 'w') as f_outer:
        f_outer.write(outer_bbox_label)

        # Save the label information for ellipse presence
    with open(save_path_ellipse_label, 'w') as f_ellipse:
        f_ellipse.write(presence_label)

def calculate_outer_bbox(center, axes, angle):
    """
    Calculate bounding box coordinates for the outer ellipse.

    Parameters:
    - center: Tuple, coordinates of the center of the outer ellipse.
    - axes: Tuple, semi-major and semi-minor axes of the outer ellipse.
    - angle: Rotation angle of the outer ellipse.

    Returns:
    - Tuple: Top-left and bottom-right coordinates of the bounding box (x1, y1, x2, y2).
    """
    # Calculate rotated bounding box for the outer ellipse
    ellipse_points = cv2.ellipse2Poly(center, tuple(axes), int(angle), 0, 360, 5)
    rect = cv2.boundingRect(ellipse_points)

    # Return top-left and bottom-right coordinates of the bounding box
    return rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]


# # Create a directory to save the images and labels
os.makedirs("try", exist_ok=True)
os.makedirs("try_labels_1", exist_ok=True)
os.makedirs("try_labels_2", exist_ok=True)

# Plot the saved image with grid using matplotlib
for i in range(1, 11):
    size = (128, 128)  # Set the size to 128x128 pixels
    center = np.random.randint(50, 64, size=(2,))  # Adjust the center range as needed
    axes = np.random.randint(20, 64, size=(2,))  # Adjust the main ellipse axes as needed
    # Ensure that the major axis is always greater than the minor axis
    if axes[0] < axes[1]:
        axes[0], axes[1] = axes[1], axes[0]

    # Adjust the ellipse parameters based on image dimensions
    center[0] = np.clip(center[0], axes[0], size[1] - axes[0])
    center[1] = np.clip(center[1], axes[1], size[0] - axes[1])
    axes[0] = min(axes[0], size[1] - center[0])
    axes[1] = min(axes[1], size[0] - center[1])

    angle = np.random.randint(0, 180)  # Adjust the rotation angle range as needed

    noisy_image, _ = create_imperfect_parent_ellipse(size, center, axes, angle)
    save_path_image = f"try/image_{i}.png"
    save_path_outer_label = f"try_labels_1/outer_bbox_label_{i}.txt"
    save_path_ellipse_label = f"try_labels_2/presence_label_{i}.txt"

    ellipse_present = 1

    # Calculate label information for outer ellipse
    outer_bbox_coords = calculate_outer_bbox(center, axes, angle)
    outer_bbox_data = f"{ellipse_present} {outer_bbox_coords[0]} {outer_bbox_coords[1]} {outer_bbox_coords[2]} {outer_bbox_coords[3]}"
    outer_bbox_label = outer_bbox_data

    # Calculate label information for ellipse presence
    ellipse_label_data = f"{ellipse_present} {center[0]} {center[1]} {angle} {axes[0]} {axes[1]}"
    ellipse_label = ellipse_label_data


    if all([i >= 0 for i in outer_bbox_coords]):
        # Save the image only if all outer box labels are positive
        save_image_with_labels(noisy_image, ellipse_label, outer_bbox_label, save_path_image, save_path_outer_label, save_path_ellipse_label)
