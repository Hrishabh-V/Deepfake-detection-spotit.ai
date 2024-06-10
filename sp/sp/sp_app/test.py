from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the previously saved model
previous_model_path = r"C:\Users\SHAHEEM\PycharmProjects\sp\sp_app\meso_inception4_model_march (1).h5"
loaded_model = load_model(previous_model_path)

# Function to visualize feature maps, activated neurons, and original image
def visualize_feature_maps_with_image(model, img_array):
    # Get the output of the last convolutional layer
    last_conv_layer = model.get_layer('conv2d_15')  # Using the correct layer name
    last_conv_output = Model(inputs=model.input, outputs=last_conv_layer.output)

    # Compute feature maps
    feature_maps = last_conv_output.predict(img_array)

    # Plot original image
    # Uncomment and add plotting code here

    # Plot each feature map
    # Uncomment and add plotting code here

# Function to predict using the loaded model
def imagepredict(test_image_path):
    img = image.load_img(test_image_path, target_size=(256, 256))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Preprocess the input image
    img_array /= 255.0

    # Visualize feature maps with original image using the loaded model
    visualize_feature_maps_with_image(loaded_model, img_array)

    # Perform prediction
    prediction = loaded_model.predict(img_array)


    pre=prediction
    if prediction >= 0.5:
        result = "Deepfake"
    else:
        pre=1-pre
        result = "Real"

    return pre, result

def imagepredict(test_image_path):
    img = image.load_img(test_image_path, target_size=(256, 256))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Preprocess the input image
    img_array /= 255.0

    # Visualize feature maps with original image using the loaded model
    visualize_feature_maps_with_image(loaded_model, img_array)

    # Perform prediction
    prediction = loaded_model.predict(img_array)


    pre=prediction
    if prediction >= 0.5:
        result = "Deepfake"
    else:
        pre=1-pre
        result = "Real"

    return pre, result

# Example usage
# prediction, result = imagepredict(r"C:\Users\SHAHEEM\Downloads\r1.jpg")
# print("Prediction Probability:", prediction)
# print("Result:", result)
