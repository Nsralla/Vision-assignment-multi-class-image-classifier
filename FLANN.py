import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
from sklearn.cluster import KMeans

# -----------------------------------------------------
# STEP1: LOAD DATA SET
training_images_path = 'training_images'
query_path = 'testing_images'

# READ training IMAGES NAMES
training_images_names = []
for filename in os.listdir(training_images_path):
    if filename.endswith('.png'):
        training_images_names.append(os.path.join(training_images_path, filename))
        
# READ query IMAGES NAMES
query_images_names = []
for filename in os.listdir(query_path):
    if filename.endswith('.png'):
        query_images_names.append(os.path.join(query_path, filename))

print(f'Number of training images: {len(training_images_names)}')
print(f'Number of query images: {len(query_images_names)}')

# -----------------------------------------------------
# STEP2: IMAGE PREPROCESSING
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Image not found: {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

preprocessed_training_images = []
for image_path in training_images_names:
    img = preprocess_image(image_path)
    if img is not None:
        preprocessed_training_images.append(img)

preprocessed_query_images = []
for image_path in query_images_names:
    img = preprocess_image(image_path)
    if img is not None:
        preprocessed_query_images.append(img)

print(f'Preprocessed training images: {len(preprocessed_training_images)}')
print(f'Preprocessed query images: {len(preprocessed_query_images)}')

# -----------------------------------------------------
# STEP 3: FEATURE EXTRACTION (SIFT)
sift = cv2.SIFT_create()

start_feature_extraction = time.time()

all_descriptors = []
image_descriptors = {}
image_keypoints = {}

for idx, img in enumerate(preprocessed_training_images):
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        all_descriptors.extend(descriptors)  # Flatten descriptors
        image_descriptors[idx] = descriptors
        image_keypoints[idx] = keypoints
    else:
        print(f'Error finding descriptor for image {training_images_names[idx]}')

# For queries, store keypoints and descriptors
query_keypoints_list = []
query_descriptors_list = []
for qimg in preprocessed_query_images:
    qkp, qdesc = sift.detectAndCompute(qimg, None)
    if qdesc is not None:
        query_keypoints_list.append(qkp)
        query_descriptors_list.append(qdesc)
    else:
        query_keypoints_list.append([])
        query_descriptors_list.append(None)

end_feature_extraction = time.time()
feature_extraction_time = end_feature_extraction - start_feature_extraction
print(f"Feature extraction time: {feature_extraction_time:.4f} seconds")

# -----------------------------------------------------
# STEP 4: BUILD CODEBOOK (VISUAL VOCABULARY) USING K-MEANS
all_descriptors_array = np.array(all_descriptors)
k = 900
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(all_descriptors_array)
visual_words = kmeans.cluster_centers_

print(f"visual words shape: {visual_words.shape}")
print(f"first visual word: {visual_words[0]}")

# -----------------------------------------------------
# STEP 5: Represent Images as histograms of visual words
def quantize_descriptors(descriptors, kmeans_model):
    cluster_indices = kmeans_model.predict(descriptors)
    return cluster_indices

image_histograms = {}
for idx, descriptors in image_descriptors.items():
    clusters = quantize_descriptors(descriptors, kmeans)
    hist, _ = np.histogram(clusters, bins=k, range=(0,k))
    hist = hist / np.sum(hist)  # Normalize histogram
    image_histograms[idx] = hist

# -----------------------------------------------------
# STEP 6: Compute TF-IDF weights
N = len(image_histograms)  # number of images
df = np.zeros(k)
# find number of images containing each feature
for hist in image_histograms.values():
    df += hist > 0

idf = np.log(N / (1 + df))  # Adding 1 to avoid division by zero

tf_idf_histograms = {}
for image_id, hist in image_histograms.items():
    tf_idf = idf * hist
    tf_idf_histograms[image_id] = tf_idf

# -----------------------------------------------------
# STEP 7: BUILD INVERTED FILE INDEX
inverted_index = {word_id: [] for word_id in range(k)}
for image_id, tf_idf_hist in tf_idf_histograms.items():
    tf_hist = image_histograms[image_id]
    for word_id in np.where(tf_hist > 0 )[0]:
        weight = tf_idf_hist[word_id]
        inverted_index[word_id].append({'image_id': image_id, 'weight': weight})

# -----------------------------------------------------
# STEP 8: IMAGE MATCHING USING FLANN-BASED MATCHER

# Initialize FLANN-Based Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)

query_tf_idf_histograms = []
for qdesc in query_descriptors_list:
    if qdesc is not None:
        cluster_indices = quantize_descriptors(qdesc, kmeans)
        hist, _ = np.histogram(cluster_indices, bins=k, range=(0, k))
        tf_hist = hist.astype(float) / np.sum(hist)  # Normalize histogram
        tf_idf_hist = tf_hist * idf
        query_tf_idf_histograms.append(tf_idf_hist)
    else:
        query_tf_idf_histograms.append(np.zeros(k))

def compute_cosine_similarity(vector1, vector2):
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 0.0
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

all_results = []
for idx, query_hist in enumerate(query_tf_idf_histograms):
    # Retrieve candidate images
    candidate_images = set()
    for word_id in np.where(query_hist > 0)[0]:
        candidates = inverted_index.get(word_id, [])
        candidate_images.update([item['image_id'] for item in candidates])
    
    # Compute similarities
    similarities = []
    for image_id in candidate_images:
        candidate_hist = tf_idf_histograms[image_id]
        sim = compute_cosine_similarity(query_hist, candidate_hist)
        similarities.append((image_id, sim))
    
    # Sort candidates by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    query_image_name = query_images_names[idx] if idx < len(query_images_names) else f'Query_{idx}'
    top_matches = similarities[:5]
    result_entry = {
        'query_image': query_image_name,
        'query_index': idx,
        'matches': [(training_images_names[m[0]], m[1], m[0]) for m in top_matches]
    }
    all_results.append(result_entry)

def extract_arabic_word(image_path):
    filename = os.path.basename(image_path)  # "user050_sakhar_012.png"
    parts = filename.split('_')  # ["user050", "sakhar", "012.png"]
    if len(parts) > 1:
        arabic_word = parts[1]
        return arabic_word
    return None

correct_count = 0
total_queries = 0

with open("results.txt", "w") as f:
    for result in all_results:
        query_image = result["query_image"]
        query_word = extract_arabic_word(query_image)
        
        f.write(f'Query Image: {query_image}\n')
        f.write('Top 5 Matches:\n')
        
        top_matches = result['matches']
        if len(top_matches) > 0:
            # Consider only the top-1 match for accuracy
            top_match_image, top_match_sim, _ = top_matches[0]
            f.write(f'  1. Image Name: "{top_match_image}", Similarity: {top_match_sim:.4f}\n')
            
            # Check if top match has the same Arabic word
            top_match_word = extract_arabic_word(top_match_image)
            
            # Only evaluate accuracy if both query and top match words were extracted
            if query_word is not None and top_match_word is not None:
                total_queries += 1
                if query_word == top_match_word:
                    correct_count += 1
            
            # Write the rest of the matches (2nd to 5th)
            for rank, (img_name, sim, _) in enumerate(top_matches[1:], start=2):
                f.write(f'  {rank}. Image Name: "{img_name}", Similarity: {sim:.4f}\n')
        else:
            f.write("  No Matches Found\n")
        f.write('\n')

if total_queries > 0:
    accuracy = (correct_count / total_queries) * 100
    print(f"Accuracy based on top matches: {accuracy:.2f}%")
else:
    print("No queries were evaluated for accuracy.")

# -----------------------------------------------------
# STEP 9: FEATURE MATCHING AND VISUALIZATION USING FLANN

good_matches_ratios = []
start_matching = time.time()

def resize_to_match_height(img_to_resize, reference_img):
    h_ref = reference_img.shape[0]
    h, w = img_to_resize.shape[:2]
    if h == 0:
        return img_to_resize
    new_w = int((w / h) * h_ref)
    resized_img = cv2.resize(img_to_resize, (new_w, h_ref))
    return resized_img

num_to_select = min(15, len(all_results))
selected_indices = random.sample(range(len(all_results)), num_to_select)

for idx in selected_indices:
    result = all_results[idx]
    q_idx = result["query_index"]
    qkp = query_keypoints_list[q_idx]
    qdesc = query_descriptors_list[q_idx]
    qimg_color = cv2.imread(result["query_image"])
    
    if qdesc is None or len(qkp) == 0:
        continue

    if len(result["matches"]) == 0:
        continue
    best_match_path, best_match_sim, best_match_id = result["matches"][0]
    tkp = image_keypoints[best_match_id]
    tdesc = image_descriptors[best_match_id]
    timg_color = cv2.imread(best_match_path)

    if tdesc is None or len(tkp) == 0:
        continue

    qimg_kp = cv2.drawKeypoints(qimg_color, qkp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    timg_kp = cv2.drawKeypoints(timg_color, tkp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow("Query Image with Keypoints", qimg_kp)
    cv2.imshow("Training Image with Keypoints", timg_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Use FLANN for KNN matching
    matches = flann.knnMatch(qdesc, tdesc, k=2)

    # Apply Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    
    # Optional: Geometric verification with RANSAC
    if len(good) > 4:
        points_query = np.float32([qkp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        points_train = np.float32([tkp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        homography, mask = cv2.findHomography(points_query, points_train, cv2.RANSAC, 5.0)
        if homography is not None:
            matchesMask = mask.ravel().tolist()
            good_inlier = [m for m, inlier in zip(good, matchesMask) if inlier]
            accuracy = len(good_inlier) / len(good) if len(good) > 0 else 0
        else:
            accuracy = len(good) / len(good) if len(good) > 0 else 0
    else:
        accuracy = len(good) / len(good) if len(good) > 0 else 0
    
    good_matches_ratios.append(accuracy)

    # Draw the good matches
    matched_image = cv2.drawMatches(qimg_color, qkp, timg_color, tkp, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matched Keypoints (Query vs Best Match)", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

end_matching = time.time()
matching_time = end_matching - start_matching
print(f"Feature matching time: {matching_time:.4f} seconds")

if len(good_matches_ratios) > 0:
    avg_accuracy = np.mean(good_matches_ratios) * 100
    print(f"Average Accuracy of good matches: {avg_accuracy:.2f}%")
else:
    print("No matches found to calculate accuracy.")

# -----------------------------------------------------
# STEP 10: ROBUSTNESS TESTS FOR MULTIPLE QUERIES

def add_noise(img, mean=0, var=10):
    row, col = img.shape
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, (row, col))
    noisy_img = img + gaussian
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Calculate the absolute value of cosine and sine from rotation matrix
    cos_val = np.abs(M[0, 0])
    sin_val = np.abs(M[0, 1])

    # Compute new bounding dimensions of the image so we don't lose parts after rotation
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    # Adjust the rotation matrix to account for the translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the actual rotation with the adjusted matrix and new image size
    rotated = cv2.warpAffine(img, M, (new_w, new_h))

    return rotated

def scale_image(img, scale=0.5):
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = cv2.resize(img, (new_w, new_h))
    return scaled

def change_illumination(img, gamma=1.0):
    # gamma < 1.0 -> brighter, gamma > 1.0 -> darker
    img_normalized = img / 255.0
    img_gamma = np.power(img_normalized, gamma)
    img_corrected = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)
    return img_corrected

print("\nRobustness Test Results for Multiple Queries:")

num_to_select_robustness = min(15, len(all_results))
robustness_indices = random.sample(range(len(all_results)), num_to_select_robustness)

for idx in robustness_indices:
    result = all_results[idx]

    if len(result["matches"]) == 0:
        continue

    q_idx = result["query_index"]
    test_query = preprocessed_query_images[q_idx]
    orig_qkp, orig_qdesc = query_keypoints_list[q_idx], query_descriptors_list[q_idx]
    if orig_qdesc is None or len(orig_qkp) == 0:
        continue

    best_match_path, best_match_sim, best_match_id = result["matches"][0]
    base_tkp = image_keypoints[best_match_id]
    base_tdesc = image_descriptors[best_match_id]
    base_timg_color = cv2.imread(best_match_path)

    orig_matches = flann.knnMatch(orig_qdesc, base_tdesc, k=2)
    orig_good = sum([1 for m, n in orig_matches if m.distance < 0.7 * n.distance])
    orig_accuracy = orig_good / len(orig_matches) if len(orig_matches) > 0 else 0

    orig_qimg_color = cv2.imread(result["query_image"])
    orig_qimg_kp = cv2.drawKeypoints(orig_qimg_color, orig_qkp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    base_timg_kp = cv2.drawKeypoints(base_timg_color, base_tkp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def ensure_same_height(img1, img2):
        if img1.shape[0] != img2.shape[0]:
            img2 = resize_to_match_height(img2, img1)
        return img1, img2

    orig_qimg_kp, base_timg_kp_resized = ensure_same_height(orig_qimg_kp, base_timg_kp)
    combined_original = np.hstack((orig_qimg_kp, base_timg_kp_resized))
    cv2.imshow("Original Query (left) and Best Match Training Image (right)", combined_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    robustness_results = {"original": orig_accuracy}
    print(f"\nRobustness Test for Query: {result['query_image']}")
    print(f"Best Match Training Image (from original query): {best_match_path}")

    # Scale variations
    for s in [0.5, 1.5]:
        scaled_img = scale_image(test_query, s)
        skp, sdesc = sift.detectAndCompute(scaled_img, None)
        if sdesc is not None and len(skp) > 1:
            smatches = flann.knnMatch(sdesc, base_tdesc, k=2)
            sgood = sum([1 for m, n in smatches if m.distance < 0.7 * n.distance])
            sacc = sgood / len(smatches) if len(smatches) > 0 else 0
            robustness_results[f"scale_{s}"] = sacc

            scaled_color = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2BGR)
            scaled_kp_img = cv2.drawKeypoints(scaled_color, skp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            scaled_kp_img, base_timg_kp_resized = ensure_same_height(scaled_kp_img, base_timg_kp)
            combined_scaled = np.hstack((scaled_kp_img, base_timg_kp_resized))
            cv2.imshow(f"Scaled ({s}) Query Keypoints (left) and Best Match (right)", combined_scaled)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Rotation variations
    for angle in [45, 90, 180]:
        rot_img = rotate_image(test_query, angle)
        rkp, rdesc = sift.detectAndCompute(rot_img, None)
        if rdesc is not None and len(rkp) > 1:
            rmatches = flann.knnMatch(rdesc, base_tdesc, k=2)
            rgood = sum([1 for m, n in rmatches if m.distance < 0.7 * n.distance])
            racc = rgood / len(rmatches) if len(rmatches) > 0 else 0
            robustness_results[f"rotate_{angle}"] = racc

            rot_color = cv2.cvtColor(rot_img, cv2.COLOR_GRAY2BGR)
            rot_kp_img = cv2.drawKeypoints(rot_color, rkp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            rot_kp_img, base_timg_kp_resized = ensure_same_height(rot_kp_img, base_timg_kp)
            combined_rot = np.hstack((rot_kp_img, base_timg_kp_resized))
            cv2.imshow(f"Rotated ({angle}°) Query Keypoints (left) and Best Match (right)", combined_rot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Noise addition
    noisy_img = add_noise(test_query, var=50)
    nkp, ndesc = sift.detectAndCompute(noisy_img, None)
    if ndesc is not None and len(nkp) > 1:
        nmatches = flann.knnMatch(ndesc, base_tdesc, k=2)
        ngood = sum([1 for m, n in nmatches if m.distance < 0.7 * n.distance])
        nacc = ngood / len(nmatches) if len(nmatches) > 0 else 0
        robustness_results["noise"] = nacc

        noisy_color = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2BGR)
        noisy_kp_img = cv2.drawKeypoints(noisy_color, nkp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        noisy_kp_img, base_timg_kp_resized = ensure_same_height(noisy_kp_img, base_timg_kp)
        combined_noisy = np.hstack((noisy_kp_img, base_timg_kp_resized))
        cv2.imshow("Noisy Query Keypoints (left) and Best Match (right)", combined_noisy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Illumination variations
    # Test gamma values: 0.5 (brighter) and 1.5 (darker) for illumination changes
    for gamma in [0.5, 1.5]:
        illum_img = change_illumination(test_query, gamma=gamma)
        ikp, idesc = sift.detectAndCompute(illum_img, None)
        if idesc is not None and len(ikp) > 1:
            imatches = flann.knnMatch(idesc, base_tdesc, k=2)
            igood = sum([1 for m, n in imatches if m.distance < 0.7 * n.distance])
            iacc = igood / len(imatches) if len(imatches) > 0 else 0
            robustness_results[f"illumination_{gamma}"] = iacc

            illum_color = cv2.cvtColor(illum_img, cv2.COLOR_GRAY2BGR)
            illum_kp_img = cv2.drawKeypoints(illum_color, ikp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            illum_kp_img, base_timg_kp_resized = ensure_same_height(illum_kp_img, base_timg_kp)
            cv2.imshow(f"Illumination (gamma={gamma}) Query Keypoints (left) and Best Match (right)", 
                       np.hstack((illum_kp_img, base_timg_kp_resized)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    for condition, acc in robustness_results.items():
        print(f"{condition}: {acc*100:.2f}% good matches")
