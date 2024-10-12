import os
import numpy as np
import os


import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array

# Path to your dataset folder
dataset_folder = "C:/Users/vipul/Desktop/Programming Project/MOBILENET_regression"

# Function to load and preprocess images
def load_and_preprocess_images(image_folder, label_folder, num_images):
    images = []
    labels = []

    for i in range(1, num_images + 1):

        # Initialize paths outside the conditional statements
        image_path = ""
        label_path = ""

        if image_folder == "C:/Users/vipul/Desktop/Programming Project/MOBILENET_regression\ellipse_absence_images_rgb":
            image_path = os.path.join(image_folder, f"absence_image_{i}.png")
            label_path = os.path.join(label_folder, f"absence_label_{i}.txt")
        elif image_folder == "C:/Users/vipul/Desktop/Programming Project/MOBILENET_regression\BB_rgb":
            image_path = os.path.join(image_folder, f"image_{i}.png")
            label_path = os.path.join(label_folder, f"presence_label_{i}.txt")

        # Check if the image file exists before loading
        if os.path.exists(image_path):
            # Load image
            img = load_img(image_path, target_size=(128, 128), color_mode="rgb")
            img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
            images.append(img_array)

            # Load 5 labels
            with open(label_path, "r") as label_file:
                label = np.array([float(value) for value in label_file.readline().split()])
                labels.append(label)


    return np.array(images), np.array(labels)

# Load and preprocess absence images and labels
absence_images, absence_labels = load_and_preprocess_images(
    os.path.join(dataset_folder, "ellipse_absence_images_rgb"),
    os.path.join(dataset_folder, "ellipse_absence_labels"),
    num_images=3008
)

# Load and preprocess presence images and labels
presence_images, presence_labels = load_and_preprocess_images(
    os.path.join(dataset_folder, "BB_rgb"),
    os.path.join(dataset_folder, "BB_labels_1"),
    num_images=4000
)

# Concatenate absence and presence data
all_images = np.concatenate([absence_images, presence_images], axis=0)
all_labels = np.concatenate([absence_labels, presence_labels], axis=0)

# Shuffle the data
np.random.seed(42)  # Set a random seed for reproducibility
indices = np.arange(len(all_images))
np.random.shuffle(indices)

shuffled_images = all_images[indices]
shuffled_labels = all_labels[indices]

# Split the data into training, dev, and testing sets
train_images, temp_images, train_labels, temp_labels = train_test_split(
    shuffled_images, shuffled_labels, test_size=0.2, random_state=42
)

val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42
)

# Print the shapes of training, dev, and testing sets
print("Train Images Shape:", train_images.shape)
print("Train Labels Shape:", train_labels.shape)
print("Dev Images Shape:", val_images.shape)
print("Dev Labels Shape:", val_labels.shape)
print("Test Images Shape:", test_images.shape)
print("Test Labels Shape:", test_labels.shape)


##############################################################################################################



#######################################################################
import tensorflow as tf
from keras.applications import MobileNetV2
from keras import layers, models
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping,LearningRateScheduler, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model
from keras.losses import Loss

#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Label Splitting for different tasks
train_presence_labels = train_labels[:, 0]
#train_regression_labels = (np.radians(train_labels[:, 3]) -np.pi)/np.pi
train_regression_labels = train_labels[:, 1:3]
#only center

val_presence_labels = val_labels[:, 0]
#val_regression_labels = (np.radians(val_labels[:, 3]) -np.pi)/np.pi
val_regression_labels = val_labels[:,1:3]
#only center

test_presence_labels = test_labels[:, 0]
#test_regression_labels = (np.radians(test_labels[:, 3]) -np.pi)/np.pi
test_regression_labels = test_labels[:, 1:3]
# #only center


#@tf.function
def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())



# Assuming your binary images are 128x128 and have 1 channel
input_shape = (128, 128, 3)

# Load pre-trained MobileNetV2 without top layers
# Load pre-trained ResNet50 without top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# # # Unfreeze the top layers of the MobileNetV2 model
# for layer in base_model.layers[-3:]:
#     layer.trainable = True


# Add your custom layers for ellipse detection
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),

    layers.Dense(320, activation='relu'),#Try #layers.Dense(640, activation='relu'),
    layers.BatchNormalization(),
    #layers.Dropout(0.1),  # Add dropout with a dropout rate of 0.5 (adjust as needed)
    layers.Dense(2, activation='relu')  # Output layer with 1 for presence
])


# Compile the model with the custom loss function
model.compile(optimizer=Adam(learning_rate=0.015), loss='mean_squared_error', metrics=['mae', r_squared])

# Display the model summary
model.summary()


# Define learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint_filepath = 'best_model_2_bb.keras'
model_checkpoint = ModelCheckpoint(
    checkpoint_filepath,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_format='tf'  # Save in the native Keras format
)

# Train the model using mini-batch gradient descent
model.fit(
    train_images, train_regression_labels,
    epochs=20,
    batch_size=128,
    validation_data=(val_images, val_regression_labels),
    callbacks=[reduce_lr, model_checkpoint]  # Include ReduceLROnPlateau and ModelCheckpoint callbacks
)



model = load_model('best_model_2.keras', custom_objects={"r_squared": r_squared})


# Make predictions on the test set
predictions = model.predict(test_images)

# Print predicted and true regression values
print("Predicted Values:")
print(predictions[0:20])

print("\nTrue Regression Values:")
print(test_regression_labels[0:20])

# Evaluate the model on the test set
test_loss, test_mae, test_r_squared = model.evaluate(test_images, test_regression_labels)

print(f'Test Loss: {test_loss}, Test MAE: {test_mae}, Test R Squared: {test_r_squared}')
# ##################################################################################################################################################
