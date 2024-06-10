# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import Model
# import numpy as np
#
# # Load the previously saved model
# previous_model_path = "meso_inception4_model.h5"
# loaded_model = load_model(previous_model_path)
#
# # Function to visualize feature maps, activated neurons, and original image
# def visualize_feature_maps_with_image(model, img_array):
#     # Get the output of the last convolutional layer
#     last_conv_layer = model.get_layer('conv2d_15')  # Using the correct layer name
#     last_conv_output = Model(inputs=model.input, outputs=last_conv_layer.output)
#
#     # Compute feature maps
#     feature_maps = last_conv_output.predict(img_array)
#
#     # Plot original image
#     plt.figure(figsize=(8, 8))
#     plt.subplot(5, 4, 1)
#     plt.imshow(img_array[0])
#     plt.axis('off')
#     plt.title('Original Image')
#
#     # Plot each feature map
#     for i in range(feature_maps.shape[-1]):
#         plt.subplot(5, 4, i+2)
#         plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
#         plt.axis('off')
#         plt.title(f'Feature Map {i+1}')
#
#     plt.show()
#
# # Load the image for testing
# test_image_path = r'C:\Users\SHAHEEM\PycharmProjects\sp\static\images\bg.jpg'  # Replace with the path to your test image
# img = image.load_img(test_image_path, target_size=(256, 256))
#
# # Convert the image to a numpy array
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#
# # Preprocess the input image
# img_array /= 255.0
#
# # Visualize feature maps with original image using the loaded model
# visualize_feature_maps_with_image(loaded_model, img_array)


import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np

# Load the previously saved model
previous_model_path = "meso_inception4_model.h5"
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
    print(prediction,"===========================")
    print(prediction,"===========================")
    print(prediction,"===========================")
    print(prediction)
    is_deepfake = prediction[0, 0] < 0.5

    # Plot original image
    plt.figure(figsize=(8, 8))
    plt.subplot(5, 4, 1)
    plt.imshow(img_array[0])
    plt.axis('off')
    plt.title('Original Image')

    # Plot each feature map
    for i in range(feature_maps.shape[-1]):
        plt.subplot(5, 4, i+2)
        plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
        plt.axis('off')
        plt.title(f'Feature Map {i+1}')

    # Show prediction result
    if is_deepfake:
        plt.suptitle("Prediction: Deepfake", fontsize=16, color='red')
    else:
        plt.suptitle("Prediction: Real", fontsize=16, color='green')

    plt.show()

test_image_path = r'C:\Users\SHAHEEM\PycharmProjects\sp\static\images\bg.jpg'  # Replace with the path to your test image
img = image.load_img(test_image_path, target_size=(256, 256))

# Convert the image to a numpy array
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Preprocess the input image
img_array /= 255.0

# Visualize feature maps with original image and prediction using the loaded model
visualize_feature_maps_with_image(loaded_model, img_array)