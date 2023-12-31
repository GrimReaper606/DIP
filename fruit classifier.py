import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

smoothed_image = None
def imageProcessing():
    # Input directory containing class subfolders
    input_dir = "train2"
    output_dir = "process_image2"

    #  output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_folders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]

    # Iterate through each class folder
    for class_folder in class_folders:
        class_input_dir = os.path.join(input_dir, class_folder)
        class_output_dir = os.path.join(output_dir, class_folder)

        #  class output directory if it doesn't exist
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        # List all image files in the class input directory
        image_files = os.listdir(class_input_dir)

        # Iterate through each image in the class folder
        for image_file in image_files:
            # Construct full paths for input and output images
            input_image_path = os.path.join(class_input_dir, image_file)
            output_image_path = os.path.join(class_output_dir, image_file)
            print(f"Processing image: {input_image_path}")

            # Load the image
            color_image = cv2.imread(input_image_path)

            if color_image is not None:
                # image processing operations
                gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                enhanced_image = cv2.equalizeHist(gray_image)
                smoothed_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
                #k-means
                k = 3
                pixel_values = color_image.reshape((-1, 3))
                pixel_values = np.float32(pixel_values)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                segmented_image = centers[labels.flatten()].reshape(color_image.shape)

                # Canny edge detection
                edges = cv2.Canny(segmented_image, 100, 200)

                # # Saving the processed image to the class output directory
                # cv2.imwrite(output_image_path,edges)
                # print(f"Processed image saved to: {output_image_path}")
    print("Image processing complete.")
    return smoothed_image
# paths for  train and test data
train_dir = "train2"
test_dir = "test2"

# function to extract HOG features
def extract_hog_features(image):

    features, _ = hog(smoothed_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

#  function to extract color histogram features
def extract_color_histogram(image):
    b, g, r = cv2.split(image)
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    color_hist_features = np.concatenate((hist_b, hist_g, hist_r)).flatten()
    return color_hist_features

# function to extract LBP features
def extract_lbp_features(image):

    lbp_features = local_binary_pattern(smoothed_image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp_features.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_features = hist.astype("float")
    lbp_features /= (lbp_features.sum() + 1e-8)
    return lbp_features

# function to extract GLCM features
def extract_glcm_features(image):

    glcm = graycomatrix(smoothed_image, [1], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-8))
    glcm_features = [contrast, energy, entropy]
    return glcm_features

def classification():
    global clf
    # Load train data and extract features
    X_train, y_train = [], []

    for class_dir in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_dir)
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                hog_features = extract_hog_features(image)
                color_hist_features = extract_color_histogram(image)
                lbp_features = extract_lbp_features(image)
                glcm_features = extract_glcm_features(image)
                combined_features = np.concatenate((hog_features,color_hist_features, lbp_features, glcm_features))
                X_train.append(combined_features)
                y_train.append(class_dir)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Load test data and extract features
    X_test, y_test = [], []

    for class_dir in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_dir)
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                hog_features = extract_hog_features(image)
                color_hist_features = extract_color_histogram(image)
                lbp_features = extract_lbp_features(image)
                glcm_features = extract_glcm_features(image)
                combined_features = np.concatenate((hog_features,color_hist_features,lbp_features, glcm_features))
                X_test.append(combined_features)
                y_test.append(class_dir)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=350, random_state=42)
    clf.fit(X_train, y_train)

    # Prediction on the test set
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("classification complete")
    # Display the test accuracy and classification report
    print("Testing Accuracy:", test_accuracy)
    print(classification_report(y_test, y_test_pred))

# Define a function to predict the vegetable class from an image
def predict_fruit(image):
    # Extract features from the image
    # features = extract_features(image)
    hog_features = extract_hog_features(image)
    color_hist_features = extract_color_histogram(image)
    lbp_features = extract_lbp_features(image)
    glcm_features = extract_glcm_features(image)
    combined_features = np.concatenate((hog_features, color_hist_features, lbp_features, glcm_features))

    # Predict the class using the trained classifier
    predicted_class = clf.predict([combined_features])

    return predicted_class[0]

# Title
st.title("Fruit Classifier")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Predict the class when a button is clicked
    if st.button("Predict"):
        smoothed_image = imageProcessing()
        classification()
        predicted_class = predict_fruit(image)

        # Display the original and predicted images
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Display original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Display image with predicted class label
        image_with_label = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_with_label, str(predicted_class), (10, 80), font, 5, (0, 0, 0), 2, cv2.LINE_AA)
        axes[1].imshow(cv2.cvtColor(image_with_label, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Predicted Class')
        axes[1].axis('off')

        st.pyplot(fig)