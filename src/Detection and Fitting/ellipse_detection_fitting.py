import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras import backend as K
import tensorflow as tf


# Function to preprocess and display an image
def preprocess_image(image_path):
    """
    Preprocess the input image by converting to grayscale and resizing.

    Args:
    image_path (str): Path to the input image.

    Returns:
    tuple: Preprocessed grayscale image, original image, and new shape.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = np.array(image)
    shape = image.shape
    #new_shape = (1024, int((shape[0] / shape[1]) * 1024))
    img_orig = cv2.imread(image_path)


    #Uncomment to visualize:
    """cv2.imshow("Original Image", img_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("Gray and resized", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    return image, img_orig


# Function to apply Gaussian Blur and edge detection
def detect_edges(image):
    """
    Apply Gaussian Blur and Canny edge detection to the image.

    Args:
    image (numpy.ndarray): Grayscale image.

    Returns:
    numpy.ndarray: Edge-detected image.
    """
    thresholded_image = cv2.GaussianBlur(image, (5, 5), 2)
    edges = cv2.Canny(thresholded_image, 20, 200)

    # Uncomment to visualize and Save
    """cv2.imshow('Edges', edges)
    cv2.imwrite("edge6crop.png", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    return edges


# Function to find and draw contours and bounding boxes
def find_and_draw_contours(edges):
    """
    Find and draw contours on the edge-detected image.

    Args:
    edges (numpy.ndarray): Edge-detected image.

    Returns:
    tuple: Image with contours
    """
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(edges)
    total_length = 0
    num_contours = 0
    bounding_boxes = []

    #THE BLOCK BELOW CALCULATES AVRAGE CONTOUR LENGTH TO FILTER OUT NON IMPORATANT CONTOURS,
    #TEMPORARRILY TURNED OFF TO CHECK HOW NEURAL NETWORK PERFORMS WITHOUT ANY PREPROCESSING OF CONTOURS
    # for i, hierarchy_entry in enumerate(hierarchy[0]):
    #     if hierarchy_entry[3] != -1:  # Check if the contour has a parent
    #         parent_contour = contours[hierarchy_entry[3]]
    #         child_contour = contours[i]
    #         parent_length = cv2.arcLength(parent_contour, True)
    #         child_length = cv2.arcLength(child_contour, True)
    #
    #         total_length += parent_length + child_length
    #         num_contours += 2
    #
    # average_length = total_length / num_contours if num_contours > 0 else 0

    for i, hierarchy_entry in enumerate(hierarchy[0]):
        if hierarchy_entry[3] != -1:
            parent_contour = contours[hierarchy_entry[3]]
            child_contour = contours[i]

            # Draw the child contour
            cv2.drawContours(contour_image, [child_contour], -1, 255, 1)
            x, y, w, h = cv2.boundingRect(child_contour)
            # bounding_boxes.append((x, y, w, h))

            # Draw the parent contour
            # if parent_length >= average_length /4 and parent_length <= average_length*4:
            cv2.drawContours(contour_image, [parent_contour], -1, 255, 1)
            #x, y, w, h = cv2.boundingRect(parent_contour)
            # bounding_boxes.append((x, y, w, h))

    # Uncomment to visualize and Save
    """cv2.imshow("Contours", contour_image)
    cv2.imwrite("contour6crop.png", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    return contour_image


# Function to draw bounding boxes on an image
def draw_bounding_boxes(contour_image):
    """
    Draw bounding boxes around the detected contours.

    Args:
    Contour Image (numpy.ndarray)
    bounding_boxes (list): List of bounding boxes.

    Returns:
    bounding_boxes (list): List of bounding Boxes of all the contours
    """
    contours = contour_image.copy()
    BB = [] #List to store Bounding Boxes of the contours

    # Find contours in Preprocessed Image
    c, _ = cv2.findContours(contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in c:
        x, y, w, h = cv2.boundingRect(contour)
        BB.append((x, y, w, h))

    sorted_BB = sorted(BB, key=lambda box: 2 * (box[2] + box[3]), reverse=True)
    filtered_BB = sorted_BB[:]

    for box in filtered_BB:
        x, y, w, h = box
        cv2.rectangle(contours, (x, y), (x + w, y + h), 255, 3)

    # Uncomment to visualize and Save
    """
    cv2.imwrite("box6crop.png", contour_image)
    cv2.imshow('Image with Boxes', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    return filtered_BB


# Function to calculate the F1 score
def f1_score(y_true, y_pred):
    """
    Calculate the F1 score for model evaluation.

    Args:
    y_true (tensor): True labels.
    y_pred (tensor): Predicted labels.

    Returns:
    tensor: F1 score.
    """
    y_pred = K.round(y_pred)
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')

    true_positives = K.sum(K.round(y_true * y_pred))
    possible_positives = K.sum(K.round(y_true))
    predicted_positives = K.sum(K.round(y_pred))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1


# Function to prepare image batches for model prediction
def prepare_batch_for_prediction(filtered_BB, image):
    """
    Prepare batches of cropped images for model prediction.

    Args:
    filtered_BB (list): List of filtered bounding boxes.
    image (numpy.ndarray): Original grayscale image.

    Returns:
    numpy.ndarray: Batch of preprocessed images ready for prediction.
    """
    batch = []

    for box in filtered_BB:
        x, y, w, h = box

        # Crop the region from the original image before resizing
        cropped_img = image[y - 2:y + h + 2, x - 2:x + w + 2]

        # Resize the cropped image
        larger_dim = max(cropped_img.shape[0], cropped_img.shape[1])
        if larger_dim > 128:
            resized_img = cv2.resize(cropped_img, (
                int(128 * cropped_img.shape[1] / larger_dim), int(128 * cropped_img.shape[0] / larger_dim)))
        else:
            resized_img = cropped_img

        # Create a black background
        black_background = np.zeros((128, 128), dtype=np.uint8)

        # Calculate the position to paste the resized image on the black background
        paste_x = (128 - resized_img.shape[1]) // 2
        paste_y = (128 - resized_img.shape[0]) // 2

        # Paste the resized image onto the black background
        black_background[paste_y:paste_y + resized_img.shape[0], paste_x:paste_x + resized_img.shape[1]] = resized_img
        resized_img = np.stack((black_background, black_background, black_background), axis=-1)
        resized_img = resized_img / 255.0

        # Reshape the image to match the model input shape
        resized_img = np.reshape(resized_img, (1, 128, 128, 3))

        # Append to the batch
        batch.append(resized_img)

    batch_array = np.vstack(batch)
    return batch_array


# Function to load the model and make predictions
def load_model_and_predict(batch_array):
    """
    Load the pre-trained model and make predictions on the input batch.

    Args:
    batch_array (numpy.ndarray): Batch of preprocessed images.

    Returns:
    numpy.ndarray: Predictions made by the model.
    """
    #LOAD Pretrained model
    model = load_model('best_model_2.keras', custom_objects={"f1_score": f1_score})

    # Make predictions using the loaded model
    predictions = model.predict(batch_array)
    return predictions


# Function to filter bounding boxes based on model predictions
def filter_bounding_boxes(predictions, filtered_BB, threshold=0.5):
    """
    Filter bounding boxes based on model predictions.

    Args:
    predictions (numpy.ndarray): Predictions made by the model.
    filtered_BB (list): List of filtered bounding boxes.
    threshold (float): Threshold for filtering predictions.

    Returns:
    list: Filtered bounding boxes.
    """

    # Filter bounding boxes based on the prediction threshold (0.5 in this example)
    filtered_boxes = [box for box, prediction in zip(filtered_BB, predictions) if prediction >= threshold]
    return filtered_boxes


# Function to create the final result image
def create_result_image(contour_image, filtered_BB, filtered_boxes):
    """
    Create the final result image by applying filters based on the bounding boxes.

    Args:
    contour_image (numpy.ndarray): Contour image.
    filtered_BB (list): List of filtered bounding boxes.
    filtered_boxes (list): List of bounding boxes after applying model predictions.

    Returns:
    numpy.ndarray: Final result image.
    """
    result_image = np.zeros_like(contour_image)

    for box in filtered_BB:
        if box in filtered_boxes:
            x, y, w, h = box
            result_image[y:y + h, x:x + w] = contour_image[y:y + h, x:x + w]

    # Uncomment to visualize and Save
    """cv2.imwrite("final6.png", result_image)
    cv2.imshow('Final Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    return result_image


# Function to fit ellipses to contours and find their centroids
def fit_ellipses_to_contours(prediction_image, colored_image):
    """
    Fit ellipses to contours and find their centroids.

    Args:
    prediction_image (numpy.ndarray): CNN filtered Contour image.
    colored_image (numpy.ndarray): Original colored image.

    Returns:
    tuple: Image with ellipses and centroids, list of centroids.
    """
    contours, hierarchy = cv2.findContours(prediction_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Convert the contours to a list of numpy arrays
    contour_list = [np.squeeze(contour) for contour in contours]
    found_centers = []

    for contour in contour_list:
        if len(contour) >= 5:  # Fit an ellipse only if there are more than 5 points
            # Fit the ellipse
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(colored_image, ellipse, (0, 0, 255), 3)

            x, y, w, h = cv2.boundingRect(contour)

            # Create a mask for the bounding box
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            cv2.drawContours(mask, [contour], -1, 0, thickness=3)

            # Apply Otsu's thresholding only within the bounding box
            bb_image = image[y:y + h, x:x + w]  # Assuming a single channel image
            _, otsu_thresholded_bb = cv2.threshold(bb_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Zero out points outside the fitted ellipse within the bounding box
            otsu_thresholded_bb = cv2.bitwise_and(otsu_thresholded_bb, mask[y:y + h, x:x + w])

            # # Update the original image with the thresholded bounding box
            # image[y:y + h, x:x + w] = otsu_thresholded_bb
            ################################################################
            #GETTING CETROIDS OF CCC_MAKRKERS BY MOMENTS METHOD
            # Centroid of bounding boxes
            # Calculate moments and centroid within the bounding box
            roi = otsu_thresholded_bb #image[y:y + h, x:x + w]
            M = cv2.moments(roi)
            centroid_x = int(M["m10"] / (M["m00"] + 1e-20)) + x  # Adjust centroid coordinates based on the bounding box
            centroid_y = int(M["m01"] / (M["m00"] + 1e-20)) + y
            found_centers.append((centroid_x, centroid_y))

            cv2.circle(colored_image, (centroid_x, centroid_y), 2, (0, 255, 0), 3)

            #####################################################################################
            #GETTING DIRECVT CENTERS OF THE FITTED ELLIPSE
            # Extract ellipse parameters
            center, axes, angle = ellipse
            center = tuple(map(int, center))
            major_axis_endpoint1 = (int(center[0] + 0.5 * axes[0] * np.cos(np.radians(angle))),
                                    int(center[1] + 0.5 * axes[0] * np.sin(np.radians(angle))))
            major_axis_endpoint2 = (int(center[0] - 0.5 * axes[0] * np.cos(np.radians(angle))),
                                    int(center[1] - 0.5 * axes[0] * np.sin(np.radians(angle))))
            # Draw the center
            cv2.circle(colored_image, center, 2, (0, 0, 255), -1)
    return colored_image, found_centers


# Main script
if __name__ == "__main__":
    image_path = 'input6crop.png'

    # Step 1: Preprocess the image
    image, img_orig = preprocess_image(image_path)

    # Step 2: Detect edges
    edges = detect_edges(image)

    # Step 3: Find and draw contours
    contour_image = find_and_draw_contours(edges)

    # Step 4: Draw bounding boxes
    filtered_BB = draw_bounding_boxes(contour_image)

    # Step 5: Prepare batches for prediction
    batch_array = prepare_batch_for_prediction(filtered_BB, image)

    # Step 6: Load model and predict
    predictions = load_model_and_predict(batch_array)

    # Step 7: Filter bounding boxes
    filtered_boxes = filter_bounding_boxes(predictions, filtered_BB, threshold=0.5)

    # Step 8: Create result image
    result_image = create_result_image(contour_image, filtered_BB, filtered_boxes)

    # Step 9: Fit ellipses to contours and find centroids
    #colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ccc_markers, centers = fit_ellipses_to_contours(result_image, img_orig)

    cv2.imwrite("NN6ellipse.png", ccc_markers)