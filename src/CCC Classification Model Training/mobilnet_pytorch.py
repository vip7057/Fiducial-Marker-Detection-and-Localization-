import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torchvision.models import mobilenet_v2

# Define dataset folder
dataset_folder = "C:/Users/vipul/Desktop/Programming Project/DATASET"

# Custom dataset class for PyTorch
class CustomDataset(Dataset):
    def __init__(self, image_folder, label_folder, num_images, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.num_images = num_images
        self.transform = transform
        self.images = []
        self.labels = []

        # Load images and labels
        self.load_images_and_labels()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def load_images_and_labels(self):
        for i in range(1, self.num_images + 1):
            # Initialize paths outside the conditional statements
            image_path = ""
            label_path = ""

            if self.image_folder.endswith("ellipse_absence_images_rgb"):
                image_path = os.path.join(self.image_folder, f"absence_image_{i}.png")
                label_path = os.path.join(self.label_folder, f"absence_label_{i}.txt")
            elif self.image_folder.endswith("ellipse_presence_images_rgb"):
                image_path = os.path.join(self.image_folder, f"presence_image_{i}.png")
                label_path = os.path.join(self.label_folder, f"presence_label_{i}.txt")

            # Load image
            image = Image.open(image_path).convert("RGB")
            self.images.append(image)

            # Load label
            with open(label_path, "r") as label_file:
                first_value = float(label_file.readline().split()[0])
                self.labels.append(first_value)

# Define transforms
transform = ToTensor()

# Create datasets
absence_dataset = CustomDataset(
    image_folder=os.path.join(dataset_folder, "ellipse_absence_images_rgb"),
    label_folder=os.path.join(dataset_folder, "ellipse_absence_labels"),
    num_images=3008,
    transform=transform
)

presence_dataset = CustomDataset(
    image_folder=os.path.join(dataset_folder, "ellipse_presence_images_rgb"),
    label_folder=os.path.join(dataset_folder, "ellipse_presence_labels"),
    num_images=3712,
    transform=transform
)

# Concatenate datasets
all_dataset = torch.utils.data.ConcatDataset([absence_dataset, presence_dataset])

# Split the dataset into training, dev, and test sets
train_size = int(0.7 * len(all_dataset))
dev_size = test_size = (len(all_dataset) - train_size) // 2
train_dataset, dev_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, dev_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print the shapes of training, dev, and testing sets
print("Train Dataset Length:", len(train_dataset))
print("Dev Dataset Length:", len(dev_dataset))
print("Test Dataset Length:", len(test_dataset))
#################################################################################################################

# Define custom F1 score function
def f1_score_custom(y_true, y_pred):
    epsilon = 1e-7
    true_positives = (y_true * y_pred).sum(dim=0)
    predicted_positives = y_pred.sum(dim=0)
    possible_positives = y_true.sum(dim=0)
    precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (possible_positives + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1.mean()

###############################################################################################
# Assuming your binary images are resized to 128x128
input_size = 128
input_channels = 3

# Load pre-trained MobileNetV2 without top layers
base_model = mobilenet_v2(pretrained=True)
base_model = nn.Sequential(*list(base_model.children())[:-1])  # Remove last fully connected layer

# Freeze the pre-trained layers
for param in base_model.parameters():
    param.requires_grad = False


# Add custom layers for ellipse detection
class CustomModel(nn.Module):
    def __init__(self, base_model):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1280, 1)
        # self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        x = self.pooling(x).view(x.size(0), -1)
        # x = nn.functional.relu(self.fc1(x))
        x = self.sigmoid(self.fc1(x))
        return x


model = CustomModel(base_model)

# Check if CUDA (GPU) is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Display the model summary
print(model)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define data loaders for training, validation, and test sets
# Assuming you have already loaded your dataset into train_dataset, dev_dataset, and test_dataset

# Assuming you have defined your own Dataset class for loading data, otherwise, replace Dataset with appropriate dataset class
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
epochs = 5
best_loss = float('inf')
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    train_predictions = []
    train_true_labels = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * inputs.size(0)

        train_predictions.extend(outputs.detach().cpu().numpy())
        train_true_labels.extend(labels.cpu().numpy())

    total_train_loss /= len(train_loader.dataset)
    train_predictions = np.concatenate(train_predictions)
    train_predictions = torch.sigmoid(torch.FloatTensor(train_predictions)).numpy()
    train_true_labels = np.array(train_true_labels)
    train_f1 = f1_score_custom(torch.FloatTensor(train_true_labels), torch.FloatTensor(train_predictions)).item()

    # Evaluation on the validation set
    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_true_labels = []
    with torch.no_grad():
        for inputs, labels in dev_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            val_loss += loss.item() * inputs.size(0)

            val_predictions.extend(outputs.detach().cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())

    val_loss /= len(dev_loader.dataset)
    val_predictions = np.concatenate(val_predictions)
    val_predictions = torch.sigmoid(torch.FloatTensor(val_predictions)).numpy()
    val_true_labels = np.array(val_true_labels)
    val_f1 = f1_score_custom(torch.FloatTensor(val_true_labels), torch.FloatTensor(val_predictions)).item()

    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {total_train_loss:.4f}, Train F1: {train_f1:.4f}, '
          f'Validation Loss: {val_loss:.4f}, Validation F1: {val_f1:.4f}')

    # Save the model if validation loss has decreased
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# Load the saved model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Evaluate the model on the test set
test_loss = 0.0
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        test_loss += loss.item() * inputs.size(0)

        predictions.extend(outputs.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

test_loss /= len(test_loader.dataset)
predictions = np.concatenate(predictions)  # Concatenate predictions into a single NumPy array
predictions = torch.sigmoid(torch.FloatTensor(predictions)).numpy()
true_labels = np.array(true_labels)

test_accuracy = accuracy_score(true_labels, predictions.round())
test_precision = precision_score(true_labels, predictions.round())
test_recall = recall_score(true_labels, predictions.round())
test_f1 = f1_score_custom(torch.FloatTensor(true_labels), torch.FloatTensor(predictions)).item()

print(
    f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test Precision: {test_precision}, Test Recall: {test_recall}, Test F1 Score: {test_f1}')
