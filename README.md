Arabic Handwritten Text Identification Using Local Feature Extraction Techniques
1. Objective
The goal of this assignment is to explore and apply local feature extraction techniques for identifying
handwritten text. Students will implement and compare SIFT algorithm with other related algorithm
like SURF (Speeded-Up Robust Features). The evaluation will be based on accuracy, efficiency, and
robustness to variations in the handwritten samples.
2. Introduction
Feature extraction is a critical step in computer vision tasks, enabling systems to identify and
differentiate objects or patterns effectively. In handwritten text identification, local feature extraction
techniques like SIFT play a pivotal role by detecting and describing key points in images that are
invariant to scale, rotation, and illumination. This assignment focuses on the practical application of
these algorithms for distinguishing between different handwritten text samples. You will preprocess
the input data, extract local features, match key points between images, and evaluate the
performance of the methods on a given dataset. The assignment emphasizes the importance of
choosing the right features and understanding the trade-offs between accuracy and computational
efficiency.
3. Dataset
The AHAWP dataset (Arabic Handwritten Automatic Word Processing) is a comprehensive benchmark
dataset designed to aid the development and evaluation of handwritten Arabic text recognition and
identification systems. For this assignment, the focus will be exclusively on the word-level data,
allowing for targeted feature extraction and robust comparisons. The dataset includes 10 unique
Arabic words, handwritten by 82 individuals, with each writer contributing 10 samples per word. This
results in a total of 8,144 word images. Each writer is uniquely but anonymously identified by a user
ID, ensuring that samples can be tracked without revealing personal information. The diversity in
handwriting styles across writers and the consistent structure of the dataset make it an excellent
resource for exploring local feature extraction techniques and evaluating their performance.
Dataset Link: https://data.mendeley.com/datasets/2h76672znt/1/files/9031138a-b812-433e-a704-
8acb1707936e
4. Metrics for Comparison
To evaluate the performance of the algorithms, consider the following metrics:
1. Accuracy: Measure the percentage of correctly matched key points in test cases.
2. Time Efficiency: Record the execution time for feature extraction and matching for each
method.
3. Robustness: Test how well each method handles variations in scale, rotation, illumination, and
noise.
4. Number of Key Points: Compare the number of key points detected by each algorithm.
