'''import cv2
import numpy as np
import os


# Function to create a binary image with random noise
def create_random_noise_image(size):
    return np.random.randint(0, 256, size, dtype=np.uint8)


# Function to add salt noise to an image
def add_salt_noise(image, salt_prob):
    noisy_image = np.copy(image)
    salt_noise = np.random.rand(*image.shape) < salt_prob
    noisy_image[salt_noise] = 255  # Set pixels to white (255) where salt noise is present
    return noisy_image


# Function to create a binary image with various geometric shapes (edges only)
def create_geometric_shapes_image(size):
    image = np.zeros(size, dtype=np.uint8)

    # Draw various geometric shapes
    for _ in range(np.random.randint(5, 10)):  # Adjust the number of shapes
        shape_type = np.random.choice(['rectangle', 'triangle'])
        color = 255  # White color

        if shape_type == 'rectangle':
            pt1 = tuple(np.random.randint(0, size[0], size=(2,)))
            pt2 = tuple(np.random.randint(pt1, size[0], size=(2,)))
            cv2.rectangle(image, pt1, pt2, color, 1)  # 1 thickness draws only the edges

        elif shape_type == 'triangle':
            pts = np.random.randint(0, size[0], size=(3, 2))
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=1)  # 1 thickness draws only the edges

    return image


# Function to create a binary image with random lines and curves (edges only)
def create_lines_and_curves_image(size):
    image = np.zeros(size, dtype=np.uint8)

    for _ in range(np.random.randint(5, 10)):  # Adjust the number of lines and curves
        color = 255  # White color

        if np.random.rand() < 0.5:  # Draw a line
            pt1 = tuple(np.random.randint(0, size[0], size=(2,)))
            pt2 = tuple(np.random.randint(0, size[0], size=(2,)))
            cv2.line(image, pt1, pt2, color, 1)  # 1 thickness draws only the edges

        else:  # Draw a curve
            pts = np.random.randint(0, size[0], size=(3, 2))
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=False, color=color, thickness=1)  # 1 thickness draws only the edges

    return image


# Function to create a binary image with textures and patterns (edges only)
def create_textures_and_patterns_image(size):
    image = np.zeros(size, dtype=np.uint8)

    for _ in range(np.random.randint(3, 6)):  # Adjust the number of textures and patterns
        color = 255  # White color
        pt1 = tuple(np.random.randint(0, size[0], size=(2,)))
        pt2 = tuple(np.random.randint(0, size[0], size=(2,)))
        cv2.rectangle(image, pt1, pt2, color, 1)  # 1 thickness draws only the edges

    return image


# Function to create a binary image with a combination of elements (edges only)
def create_combined_elements_image(size):
    image = np.zeros(size, dtype=np.uint8)

    # Randomly choose elements to combine
    elements = [create_geometric_shapes_image, create_lines_and_curves_image, create_textures_and_patterns_image]

    for _ in range(np.random.randint(3, 6)):  # Adjust the number of combined elements
        element_type = np.random.choice(elements)
        element_image = element_type(size)
        image = cv2.bitwise_or(image, element_image)

    return image


# Function to create labels for ellipse absence images
def create_labels_absence(size):
    return f"0 0 0 0 0 0"  # Set all labels to zero


# Function to save the binary image and corresponding label
def save_image_and_label(image_data, save_path_image, save_path_label):
    cv2.imwrite(save_path_image, image_data)
    label_data = create_labels_absence(size)
    with open(save_path_label, 'w') as label_file:
        label_file.write(label_data)


# Directory to save images and labels
os.makedirs("ellipse_absence_images", exist_ok=True)
os.makedirs("ellipse_absence_labels", exist_ok=True)

# Number of ellipse absence images to generate
num_images = 761
salt_noise_prob = 0.3  # Adjust the probability of salt noise as needed

for i in range(561,num_images):
    size = (128, 128)  # Adjust the size as needed

    # Choose one of the creation functions randomly
    creation_function = np.random.choice([
        create_random_noise_image,
        # create_geometric_shapes_image,
        # create_lines_and_curves_image,
        create_textures_and_patterns_image,
        #create_combined_elements_image
    ])

    image = creation_function(size)

# Optionally, add salt noise based on the specified probability
    if np.random.rand() < salt_noise_prob:
        image_with_salt = add_salt_noise(image.copy(), 0.05)  # Adjust the salt noise intensity if needed

        # Save the image with salt noise
        save_path_image_salt = f"ellipse_absence_images/absence_image_{i}.png"
        save_path_label_salt = f"ellipse_absence_labels/absence_label_{1}.txt"
        save_image_and_label(image_with_salt, save_path_image_salt, save_path_label_salt)
    else:
        # Save the clean image without salt noise
        save_path_image_clean = f"ellipse_absence_images/absence_image_{i}.png"
        save_path_label_clean = f"ellipse_absence_labels/absence_label_{1}.txt"
        save_image_and_label(image, save_path_image_clean, save_path_label_clean)
'''
##########################################################################################################################
'''
import cv2

##SINGLE POLYGON WITH/WO NOISE##
import numpy as np
import os


# Function to create a binary image with random noise
def create_random_noise_image(size):
    return np.random.randint(0, 256, size, dtype=np.uint8)

# Function to create a binary image with one polygon (edges only)
def create_one_polygon_image(size):
    image = np.zeros(size, dtype=np.uint8)
    color = 255  # White color

    # Choose a random number of sides for the polygon (between 3 and 8)
    num_sides = np.random.randint(3, 9)

    # Generate random vertices for the polygon
    pts = np.random.randint(0, size[0], size=(num_sides, 2))

    # Reshape the vertices to form a closed polygon
    pts = pts.reshape((-1, 1, 2))

    # Draw the polygon
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=1)  # 1 thickness draws only the edges

    return image

# Function to add salt noise to an image
def add_salt_noise(image, salt_prob):
    noisy_image = np.copy(image)
    salt_noise = np.random.rand(*image.shape) < salt_prob
    noisy_image[salt_noise] = 255  # Set pixels to white (255) where salt noise is present
    return noisy_image

# Function to create labels for ellipse absence images
def create_labels_absence(size):
    return f"0 0 0 0 0 0"  # Set all labels to zero

# Function to save the binary image and corresponding label
def save_image_and_label(image_data, save_path_image, save_path_label):
    cv2.imwrite(save_path_image, image_data)
    label_data = create_labels_absence(size)
    with open(save_path_label, 'w') as label_file:
        label_file.write(label_data)

# Directory to save images and labels
os.makedirs("ellipse_absence_images", exist_ok=True)
os.makedirs("ellipse_absence_labels", exist_ok=True)

# Number of ellipse absence images to generate
num_images = 1661
salt_noise_prob = 0.3

for i in range(761,num_images):
    size = (128, 128)  # Adjust the size as needed

    # Choose one of the creation functions randomly
    creation_function = np.random.choice([
        #create_random_noise_image,
        create_one_polygon_image
    ])

    image = creation_function(size)

    # Optionally, add salt noise based on the specified probability
    if np.random.rand() < salt_noise_prob:
        image_with_salt = add_salt_noise(image.copy(), 0.05)  # Adjust the salt noise intensity if needed

        # Save the image with salt noise
        save_path_image_salt = f"ellipse_absence_images/absence_image_{i}.png"
        save_path_label_salt = f"ellipse_absence_labels/absence_label_{1}.txt"
        save_image_and_label(image_with_salt, save_path_image_salt, save_path_label_salt)
    else:
        # Save the clean image without salt noise
        save_path_image_clean = f"ellipse_absence_images/absence_image_{i}.png"
        save_path_label_clean = f"ellipse_absence_labels/absence_label_{1}.txt"
        save_image_and_label(image, save_path_image_clean, save_path_label_clean)



'''
#######################################################################################################################################

## RECT TRIANGLE WITH/WO NOISE
import cv2
import numpy as np
import os

# Set a fixed seed for reproducibility
np.random.seed(42)


# Function to create a binary image with random noise
def create_random_noise_image(size):
    return np.random.randint(0, 256, size, dtype=np.uint8)


# Function to create a binary image with one rectangle (edges only)
def create_one_rectangle_image(size):
    image = np.zeros(size, dtype=np.uint8)
    color = 255  # White color

    # Draw one rectangle
    pt1 = tuple(np.random.randint(0, size[0] // 2, size=(2,)))
    pt2 = tuple(np.random.randint(size[0] // 2, size[0], size=(2,)))
    cv2.rectangle(image, pt1, pt2, color, 1)  # 1 thickness draws only the edges

    return image


# Function to create a binary image with one triangle (edges only)
def create_one_triangle_image(size):
    image = np.zeros(size, dtype=np.uint8)
    color = 255  # White color

    # Draw one triangle
    pts = np.random.randint(0, size[0] // 2, size=(3, 2))
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=1)  # 1 thickness draws only the edges

    return image


# Function to add salt noise to an image
# Function to add salt noise to an image
def add_salt_noise(image, salt_prob):
    noisy_image = np.copy(image)
    salt_noise = np.random.rand(*image.shape) < salt_prob
    noisy_image[salt_noise] = 255  # Set pixels to white (255) where salt noise is present
    return noisy_image



# Function to create labels for ellipse absence images
def create_labels_absence(size):
    return f"0 0 0 0 0 0"  # Set all labels to zero


# Function to save the binary image and corresponding label
def save_image_and_label(image_data, save_path_image, save_path_label):
    #cv2.imwrite(save_path_image, image_data)
    label_data = create_labels_absence(size)
    with open(save_path_label, 'w') as label_file:
        label_file.write(label_data)


# Directory to save images and labels
# os.makedirs("ellipse_absence_images", exist_ok=True)
# os.makedirs("ellipse_absence_labels", exist_ok=True)

# Number of ellipse absence images to generate
num_images = 3009
salt_noise_prob =0.3

for i in range(2,num_images):
    size = (128, 128)  # Adjust the size as needed

    # Choose one of the creation functions randomly
    # creation_function = np.random.choice([
    #     #create_random_noise_image,
    #     create_one_rectangle_image,
    #     create_one_triangle_image
    # ])

    #image = creation_function(size)

    # if np.random.rand() < salt_noise_prob:
    #image_with_salt = add_salt_noise(image.copy(), 0.05)  # Adjust the salt noise intensity if needed

        # Save the image with salt noise
    save_path_image_salt = f"ellipse_absence_images/absence_image_{i}.png"
    save_path_label_salt = f"DATASET/ellipse_absence_labels/absence_label_{i}.txt"
    save_image_and_label(1, save_path_image_salt, save_path_label_salt)
#     else:
#         # Save the clean image without salt noise
#         save_path_image_clean = f"ellipse_absence_images/absence_image_{i}.png"
#         save_path_label_clean = f"ellipse_absence_labels/absence_label_{1}.txt"
#         save_image_and_label(image, save_path_image_clean, save_path_label_clean)
# #############################################################################################################


"""
import cv2
import numpy as np
import os

# Function to create a binary image with random closed-loop figures (edges only)
def create_random_closed_loop_image(size):
    image = np.zeros(size, dtype=np.uint8)  # Create a black image

    # Determine the number of shapes (1 to 4)
    num_shapes = np.random.randint(1, 5)

    for _ in range(num_shapes):
        if np.random.rand() < 0.5:
            # Random closed-loop figure with random number of sides between 6 and 12
            num_sides = np.random.randint(6, 13)
            shape_size = np.random.randint(20, 40)

            # Random position for the shape
            position = np.random.randint(0, size[0] - shape_size, size=(2,))

            # Calculate the vertices of the polygon
            theta = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
            x = position[0] + shape_size * np.cos(theta)
            y = position[1] + shape_size * np.sin(theta)
            vertices = np.column_stack((x, y)).astype(int)

            # Draw the edges of the polygon
            cv2.polylines(image, [vertices], isClosed=True, color=255, thickness=1)
        else:
            # Closed-loop shape starting and ending at a random point
            num_sides = np.random.randint(6, 13)
            shape_size = np.random.randint(20, 40)

            # Random starting point
            start_point = np.random.randint(shape_size, size=(2,))
            start_point = tuple(np.clip(start_point, 0, np.array(size) - 1))  # Corrected line

            # Calculate the vertices of the polygon
            theta = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
            x = start_point[0] + shape_size * np.cos(theta)
            y = start_point[1] + shape_size * np.sin(theta)
            vertices = np.column_stack((x, y)).astype(int)

            # Draw the edges of the polygon
            cv2.polylines(image, [vertices], isClosed=True, color=255, thickness=1)

    # Optionally add salt noise
    if np.random.rand() < 0.3:
        salt_noise = np.random.rand(*size) < 0.05
        image[salt_noise] = 255  # Set pixels to white (255) where salt noise is present

    return image

# Function to save the binary image
def save_image(image_data, save_path):
    cv2.imwrite(save_path, image_data)

# Function to save the label file (no ellipses in this case)
def save_label(save_path):
    with open(save_path, 'w') as f:
        f.write("0")  # Indicate that there are no ellipses in the image

# Create a directory to save the images and labels
# os.makedirs("no_ellipse_closed_loop_images", exist_ok=True)
# os.makedirs("no_ellipse_closed_loop_labels", exist_ok=True)

# Plot the saved image with grid using matplotlib
for i in range(2561, 2700):  # Adjust the range as needed
    size = (128, 128)  # Set the size to 128x128 pixels

    no_ellipse_closed_loop_image = create_random_closed_loop_image(size)
    save_path_image = f"DATASET/ellipse_absence_images/absence_image_{i}.png"
    #save_path_label = f"DATASET/ellipse_absence_images/absence_label_{i}.txt"

    save_image(no_ellipse_closed_loop_image, save_path_image)
    #save_label(save_path_label)

##############################################################################################################################
"""
"""
import cv2
import numpy as np
import os

# Function to create a binary image with random irregular closed-loop figures (edges only)
def create_random_closed_loop_image(size):
    image = np.zeros(size, dtype=np.uint8)  # Create a black image

    # Determine the number of shapes (1 to 4)
    num_shapes = np.random.randint(1, 5)

    for _ in range(num_shapes):
        # Random irregular closed-loop shape
        num_points = np.random.randint(10, 20)  # Adjust the number of points as needed
        shape_size = np.random.randint(20, 40)  # Adjust the size as needed

        # Random starting point
        start_point = np.random.randint(shape_size, size=(2,))
        start_point = tuple(np.clip(start_point, 0, np.array(size) - 1))

        # Generate random points for the shape
        points = [start_point]
        for _ in range(num_points):
            angle = np.random.uniform(0, 2 * np.pi)
            new_point = (points[-1][0] + shape_size * np.cos(angle),
                         points[-1][1] + shape_size * np.sin(angle))
            new_point = tuple(np.clip(new_point, 0, np.array(size) - 1))
            points.append(new_point)

        # Draw the edges of the shape
        cv2.polylines(image, [np.array(points, dtype=int)], isClosed=True, color=255, thickness=1)

    # Optionally add salt noise
    if np.random.rand() < 0.3:
        salt_noise = np.random.rand(*size) < 0.05
        image[salt_noise] = 255  # Set pixels to white (255) where salt noise is present

    return image

# Function to save the binary image
def save_image(image_data, save_path):
    cv2.imwrite(save_path, image_data)

# Function to save the label file (no ellipses in this case)
def save_label(save_path):
    with open(save_path, 'w') as f:
        f.write("0")  # Indicate that there are no ellipses in the image

# Create a directory to save the images and labels
#os.makedirs("no_ellipse_irregular_shapes_labels", exist_ok=True)

# Plot the saved image with grid using matplotlib
for i in range(2, 3009):  # Adjust the range as needed
    size = (128, 128)  # Set the size to 128x128 pixels

    no_ellipse_irregular_shapes_image = create_random_closed_loop_image(size)
    #save_path_image = f"DATASET/ellipse_absence_images/absence_image_{i}.png"
    save_path_label = f"no_ellipse_irregular_shapes_labels/no_ellipse_irregular_shapes_label_{i}.txt"

    #save_image(no_ellipse_irregular_shapes_image, save_path_image)
    #save_label(save_path_label)
"""