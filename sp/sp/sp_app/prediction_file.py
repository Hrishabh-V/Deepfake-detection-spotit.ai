import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Load the previously saved model
previous_model_path = "D:/PROJECT/DATASET/trainedmodels/meso_inception4_model_march.h5"
loaded_model = load_model(previous_model_path)


# Function to visualize feature maps, activated neurons, and original image with prediction
def visualize_feature_maps_with_image(model, img_array):
    # Get the output of the last convolutional layer
    last_conv_layer = model.get_layer('conv2d_15')  # Using the correct layer name
    last_conv_output = Model(inputs=model.input, outputs=last_conv_layer.output)

    # Compute feature maps
    feature_maps = last_conv_output.predict(img_array)

    # Make prediction
    prediction = loaded_model.predict(img_array)
    prediction_score = prediction[0, 0]
    threshold = 0.5  # You can adjust this threshold as needed
    is_real_prediction = prediction_score >= threshold

    # Plot original image
    plt.figure(figsize=(8, 8))
    plt.subplot(5, 4, 1)
    plt.imshow(img_array[0])
    plt.axis('off')
    plt.title('Original Image')

    # Show prediction result
    if is_real_prediction:
        plt.suptitle("Prediction: Real", fontsize=16, color='green')
        plt.figtext(0.5, 0.05, f'Confidence Score: {prediction_score:.2f}', ha='center', fontsize=12, color='green')
    else:
        plt.suptitle("Prediction: Deepfake", fontsize=16, color='red')
        plt.figtext(0.5, 0.05, f'Confidence Score: {1 - prediction_score:.2f}', ha='center', fontsize=12, color='red')

    plt.show()

# Function to process either image or video
def process_image_or_video(input_path):
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Process image
        img = image.load_img(input_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Preprocess the input image
        visualize_feature_maps_with_image(loaded_model, img_array)
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        # Process video
        output_dir = "output_frames"
        frame_count = extract_frames_from_video(input_path, output_dir)
        print(f"Extracted {frame_count} frames from the video.")
        for i in range(1, frame_count + 1):
            frame_path = os.path.join(output_dir, f"frame_{i}.jpg")
            img = image.load_img(frame_path, target_size=(256, 256))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array /= 255.0  # Preprocess the input image
            visualize_feature_maps_with_image(loaded_model, img_array)
    else:
        print("Unsupported file format.")

# Function to extract frames from a video
def extract_frames_from_video(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
        else:
            break

    cap.release()

    return frame_count

# Input path (either an image or a video)
input_path = 'D:/PROJECT/DATASET/deepfakeset/validation/df/df00233.jpg'  # Replace with the path to your image or video

# Process image or video
process_image_or_video(input_path)
