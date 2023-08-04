# Traffic-Sign-Recognition-and-Classification

1.Dataset Details:
Number of Classes: 43 (representing different traffic sign categories)
Total Images: Approximately 50,000 images
Image Size: 32x32 pixels (grayscale)
Dataset Download Link:
The GTSRB dataset can be downloaded from Kaggle using the following link:
https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

2.Preprocessing Functions (grayscale, equalize, preprocess): These functions are used to preprocess the input images before feeding them into the model. The grayscale function converts the images from color (BGR) to grayscale. The equalize function performs histogram equalization on the grayscale image to enhance contrast. Finally, the preprocess function combines these preprocessing steps and scales the pixel values to the range [0, 1].

3.ImageDataGenerator: The ImageDataGenerator is an important tool for data augmentation in deep learning. It generates augmented images on the fly during training, which helps in improving model performance and generalization. Data augmentation creates variations of the input images by applying random transformations like shifting, zooming, shearing, and rotation.

4.Data Augmentation: The ImageDataGenerator is configured with various augmentation parameters such as width_shift_range, height_shift_range, zoom_range, shear_range, and rotation_range. These parameters control the amount and types of augmentations applied to the input images during training. Augmented images are then used to augment the training dataset, creating a more diverse set of examples for the model to learn from.

5.Data Flow and Visualization: The batches variable is used to flow the augmented data in batches during training. The next function is used to retrieve the next batch of augmented data. The code then visualizes the first 15 images from the augmented batch.

6.Model Architecture:

Convolutional Layers: The model begins with three sets of Convolutional layers, each followed by a ReLU activation function. The convolutional layers are used to extract feature maps from the input images. The first set consists of 60 filters with a kernel size of (5, 5), followed by another set of 60 filters, and finally, a third set of 30 filters with a kernel size of (5, 5).
MaxPooling Layers: After each set of Convolutional layers, a MaxPooling layer with a pool size of (2, 2) is applied to reduce spatial dimensions and focus on the most salient features.
Flatten Layer: The output of the last MaxPooling layer is flattened into a 1-dimensional vector to be fed into the dense layers.
Dense Layers: There is one Dense layer with 500 neurons, which is followed by a ReLU activation function. This layer acts as a fully connected layer and learns to combine the extracted features for classification.
Dropout Layer: A Dropout layer with a rate of 0.5 is added after the Dense layer to reduce overfitting during training.
Output Layer: The final Dense layer consists of 43 neurons (equal to the number of classes in the dataset) with a softmax activation function. This layer produces the probability distribution over the 43 classes, indicating the predicted class for each input image.

7.Model Compilation:

The model is compiled using the Adam optimizer with a learning rate of 0.001. The categorical cross-entropy loss function is chosen since this is a multi-class classification problem. The metric used to evaluate the model's performance is accuracy.
