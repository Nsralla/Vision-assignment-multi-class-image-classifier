import cv2
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans as kMeans
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier

# Suppress joblib warnings
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------
# STEP 1: LOAD DATA SET
# The following paths should point to directories containing the images.
training_images_path = 'training_images'
testing_images_path = 'testing_images'

# Get the list of training and testing image file names
training_images_names = os.listdir(training_images_path)
testing_images_names = os.listdir(testing_images_path)

# Debugging statements
print("[DEBUG] Loaded training images:", training_images_names)
print("[DEBUG] Loaded testing images:", testing_images_names)
print("[DEBUG] Training images count:", len(training_images_names))
print("[DEBUG] Testing images count:", len(testing_images_names))

# -----------------------------------------------------
# STEP 2: IMAGE PREPROCESSING
def preprocess_image(img_path):
    """
    Preprocesses the image by:
    - Reading the image from the given path.
    - Converting it to grayscale.
    Returns the grayscale image if successful, otherwise None.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Image not found at {img_path}")
        return None
    # Convert to grayscale for feature extraction
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# Preprocess training images
preprocessed_training_images = []
for image_name in training_images_names:
    image_path = os.path.join(training_images_path, image_name)
    img = preprocess_image(image_path)
    if img is not None:
        preprocessed_training_images.append((image_name, img))

# Debugging statements
print(f"[DEBUG] Preprocessed {len(preprocessed_training_images)} training images successfully.")

# Preprocess testing images
preprocessed_testing_images = []
for image_name in testing_images_names:
    image_path = os.path.join(testing_images_path, image_name)
    img = preprocess_image(image_path)
    if img is not None:
        preprocessed_testing_images.append((image_name, img))

# Debugging statements
print(f"[DEBUG] Preprocessed {len(preprocessed_testing_images)} testing images successfully.")

# -----------------------------------------------------
# DATA AUGMENTATION FUNCTIONS
def apply_scaling(img, scale_factor):
    """
    Scales the image by a given factor.
    """
    height, width = img.shape
    scaled_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    return scaled_img

def apply_rotation(img, angle):
    """
    Rotates the image by a given angle (in degrees).
    """
    height, width = img.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img

def apply_illumination(img, alpha, beta):
    """
    Adjusts image illumination by controlling contrast (alpha) and brightness (beta).
    alpha: Contrast control (1.0-3.0)
    beta: Brightness control (0-100)
    """
    illuminated_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return illuminated_img

def apply_noise(img, noise_type="gaussian", mean=0, std=25):
    if noise_type == "gaussian":
        gauss = np.random.normal(mean, std, img.shape).astype('uint8')
        noisy_img = cv2.add(img, gauss)
    elif noise_type == "salt_pepper":
        s_vs_p = 0.5
        amount = 0.02
        noisy_img = img.copy()
        
        # Number of 'salt' pixels
        num_salt = np.ceil(amount * img.size * s_vs_p)
        # Generate coordinates for salt
        coords = [np.random.randint(0, dim, int(num_salt)) for dim in img.shape]
        noisy_img[coords[0], coords[1]] = 255  # Assign white pixels
        
        # Number of 'pepper' pixels
        num_pepper = np.ceil(amount * img.size * (1.0 - s_vs_p))
        # Generate coordinates for pepper
        coords = [np.random.randint(0, dim, int(num_pepper)) for dim in img.shape]
        noisy_img[coords[0], coords[1]] = 0  # Assign black pixels

    return noisy_img

def augment_image(img):
    """
    Applies a series of augmentations (scaling, rotation, illumination change, noise)
    to the given image and returns a list of augmented images.
    """
    augmented_images = []
    # Scaling variations
    for scale_factor in [0.8, 1.0, 1.2]:
        augmented_images.append(apply_scaling(img, scale_factor))

    # Rotation variations
    for angle in [-30, 0, 30]:
        augmented_images.append(apply_rotation(img, angle))

    # Illumination variations
    for alpha, beta in [(1.2, 30), (1.0, 0), (0.8, -30)]:
        augmented_images.append(apply_illumination(img, alpha, beta))

    # Noise variations
    for noise_type in ["gaussian", "salt_pepper"]:
        augmented_images.append(apply_noise(img, noise_type))

    return augmented_images

# Augment training images
augmented_training_images = []
for image_name, img in preprocessed_training_images:
    aug_imgs = augment_image(img)
    for augmented_img in aug_imgs:
        augmented_training_images.append((image_name, augmented_img))

# Debugging statements
print(f"[DEBUG] Augmented training images count: {len(augmented_training_images)}")

# Augment testing images
augmented_testing_images = []
for image_name, img in preprocessed_testing_images:
    aug_imgs = augment_image(img)
    for augmented_img in aug_imgs:
        augmented_testing_images.append((image_name, augmented_img))

# Debugging statements
print(f"[DEBUG] Augmented testing images count: {len(augmented_testing_images)}")

# -----------------------------------------------------
# STEP 3: EXTRACT FEATURES USING SIFT (or ORB)
# We will use SIFT feature extractor.

# feature_extractor = cv2.SIFT_create()
feature_extractor = cv2.ORB_create()

# Collect all descriptors from training images
all_descriptors = []
valid_training_images = []
num_of_keypoints = 0

for image_name, img in augmented_training_images:
    keypoints, descriptors = feature_extractor.detectAndCompute(img, None)
    num_of_keypoints += len(keypoints)
    if descriptors is not None:
        all_descriptors.append(descriptors)
        valid_training_images.append((image_name, img))
    else:
        print(f"[WARNING] Skipping image {image_name} due to lack of descriptors.")

print(f"[DEBUG] Number of keypoints in training with augmentation: {num_of_keypoints}")
print(f"[DEBUG] Valid training images after feature extraction: {len(valid_training_images)}")

# If no descriptors found, cannot proceed
if len(all_descriptors) == 0:
    print("[ERROR] No descriptors found in any training image.")
    exit(1)

# Flatten all descriptors into a single numpy array for clustering
training_descriptors = np.vstack(all_descriptors)
print("[DEBUG] Stacked training descriptors shape:", training_descriptors.shape)

# -----------------------------------------------------
# STEP 4: BUILD CODEBOOK (VISUAL VOCABULARY)
k = 900  # Number of clusters (visual words)
kmeans = kMeans(n_clusters=k, random_state=42)
kmeans.fit(training_descriptors)

visual_words = kmeans.cluster_centers_
print("[DEBUG] Visual words shape:", visual_words.shape)

# -----------------------------------------------------
# STEP 5: REPRESENT IMAGES AS HISTOGRAM OF VISUAL WORDS
def quantize_descriptors(descriptors, kmeans_model):
    """
    Assigns each descriptor to its nearest cluster center (visual word).
    """
    cluster_indices = kmeans_model.predict(descriptors)
    return cluster_indices

training_features = []
training_labels = []

for image_name, img in valid_training_images:
    keypoints, descriptors = feature_extractor.detectAndCompute(img, None)
    if descriptors is not None:
        clusters = quantize_descriptors(descriptors, kmeans)
        hist, _ = np.histogram(clusters, bins=np.arange(k + 1))
        hist = hist / np.sum(hist)  # Normalize histogram
        training_features.append(hist)
        # Label extraction: assumes image_name is formatted as "class_xxx.ext"
        label = image_name.split('_')[0]
        training_labels.append(label)
    else:
        print(f"[WARNING] Skipping image {image_name} due to lack of descriptors.")

print(f"[DEBUG] Histograms generated for {len(training_features)} training images.")

# Compute TF-IDF weights
N = len(training_features)  # Number of images
df = np.zeros(k)  # Document frequency for each visual word

# Calculate document frequency (how many images contain each feature)
for hist in training_features:
    df += (hist > 0)

idf = np.log((N + 1) / (1 + df)) + 1  # Smoothed IDF

tf_idf_training_features = []
for hist in training_features:
    tf_idf = hist * idf
    tf_idf_training_features.append(tf_idf)

tf_idf_training_features = np.array(tf_idf_training_features)

# Encode labels
label_encoder = LabelEncoder()
encoded_training_labels = label_encoder.fit_transform(training_labels)

print("[DEBUG] Training features shape:", tf_idf_training_features.shape)
print("[DEBUG] Training labels distribution:", pd.Series(training_labels).value_counts())

# -----------------------------------------------------
# STEP 6: FEATURE SCALING
scaler = StandardScaler()
tf_idf_training_features_scaled = scaler.fit_transform(tf_idf_training_features)

print("[DEBUG] Scaled training features shape:", tf_idf_training_features_scaled.shape)

# -----------------------------------------------------
# Check class distribution
label_counts = pd.Series(training_labels).value_counts()
print("Class Distribution in Training Data:")
print(label_counts)

# -----------------------------------------------------
# STEP 8: MODEL SELECTION & TRAINING (Here we use SVC)
clf = SVC(C=10, gamma='auto', kernel='rbf')
print("[DEBUG] SVC model initialized with C=10, gamma='auto', kernel='rbf'")

# -----------------------------------------------------
# STEP 9: PROCESS TESTING IMAGES
testing_features = []
testing_labels = []
valid_testing_images = []
num_of_keypoints = 0

for image_name, img in augmented_testing_images:
    keypoints, descriptors = feature_extractor.detectAndCompute(img, None)
    num_of_keypoints += len(keypoints)
    if descriptors is not None:
        clusters = quantize_descriptors(descriptors, kmeans)
        hist, _ = np.histogram(clusters, bins=np.arange(k + 1))
        hist = hist / np.sum(hist)
        tf_idf = hist * idf  # Apply the same IDF weights from training
        testing_features.append(tf_idf)
        label = image_name.split('_')[0]
        testing_labels.append(label)
        valid_testing_images.append((image_name, img))
    else:
        print(f"[WARNING] Skipping image {image_name} due to lack of descriptors.")

print(f"[DEBUG] Number of keypoints in testing with augmentation: {num_of_keypoints}")
print("[DEBUG] Valid testing images after feature extraction:", len(valid_testing_images))

testing_features = np.array(testing_features)

# Scale testing features using the same scaler as training
testing_features_scaled = scaler.transform(testing_features)

# Encode testing labels using the same label encoder
encoded_testing_labels = label_encoder.transform(testing_labels)
print("[DEBUG] Testing features shape:", testing_features_scaled.shape)
print("[DEBUG] Testing labels distribution:", pd.Series(testing_labels).value_counts())

# -----------------------------------------------------
# STEP 10: PREDICT AND EVALUATE
# Train the classifier on the training data
clf.fit(tf_idf_training_features_scaled, encoded_training_labels)
print("[DEBUG] Classifier trained on scaled training features.")

# Predict on the testing data
predicted_labels = clf.predict(testing_features_scaled)
print("[DEBUG] Predictions made on testing data.")

# Calculate accuracy
accuracy = accuracy_score(encoded_testing_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
