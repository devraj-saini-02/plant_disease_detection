 Plant_disease_detection
This Convolutional Neural Network (CNN) is designed to classify plant leaf images into one of 38 categories from the PlantVillage dataset. Each category represents a specific plant species and a possible disease, or a healthy condition.

The model processes input images of size 224×224×3 and consists of several convolutional layers with Batch Normalization, MaxPooling, and Dropout for improved training stability and reduced overfitting. The final layer uses a softmax activation to output probabilities for each class.

When deployed with Gradio, users can upload a plant leaf photo and instantly receive:

The plant species name

The detected disease (or “Healthy”)

The confidence score of the prediction

This tool can help farmers, researchers, and gardeners quickly identify plant health issues from leaf images, enabling early intervention and better crop management.


Dataset : https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
