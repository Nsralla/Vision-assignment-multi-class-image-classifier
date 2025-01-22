import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
user_folder = r"C:\Users\nsrha\Downloads\isolated_words_per_user\isolated_words_per_user\user082"  # Path to the user folder
count = 0
for file in os.listdir(user_folder):
    count += 1
print(f"Total number of images: {count}")




training_directory = "./training_images"
testing_directory = "./testing_images"

# Create output directories if they don't exist
os.makedirs(training_directory, exist_ok=True)
os.makedirs(testing_directory, exist_ok=True)

print(f"Processing {user_folder}...")

# Group files by unique words
word_groups = {}
for filename in os.listdir(user_folder):
    if filename.endswith(".png"):
        # Extract the word (e.g., 'abjadiyah' from 'user001_abjadiyah_031.png')
        word = filename.split('_')[1]
        word_groups.setdefault(word, []).append(filename)
# for word, files in word_groups.items():
#     print(f"Word: {word}, Files: {files}")

# Process each word group
for word, files in word_groups.items():
    # Split files into training and testing
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    
    
    
    # Move training files
    for file in train_files:
        src = os.path.join(user_folder, file)
        dest = os.path.join(training_directory, file)
        shutil.copy(src, dest)
    
    # Move testing files
    for file in test_files:
        src = os.path.join(user_folder, file)
        dest = os.path.join(testing_directory, file)
        shutil.copy(src, dest)

print(f"Finished processing {user_folder}.")
print("Data splitting complete!")
