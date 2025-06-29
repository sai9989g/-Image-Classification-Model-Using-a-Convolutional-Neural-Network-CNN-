# CIFAR-10 Image Classification with CNN

## Overview
This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The 10 classes represent airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Project Structure
- `app.py`: Main Python script containing the complete implementation
- `Test_Accuracy.png`: Training logs showing accuracy metrics
- `Plot.png`: Visualization of training and validation accuracy over epochs
- `Output.png`: Sample predictions on test images

## Key Features
- Image preprocessing including normalization and one-hot encoding
- CNN architecture with:
  - Three convolutional layers with ReLU activation
  - Two max-pooling layers
  - Two dense layers (one hidden layer and output layer)
- Adam optimizer with categorical cross-entropy loss
- Training visualization showing accuracy progression
- Model evaluation on test data

## Results
The model achieved:
- Training accuracy: 75.52%
- Validation accuracy: 71.11%
- Test accuracy: 71%

## How to Use
1. Ensure you have Python installed with required libraries:
   ```
   pip install tensorflow matplotlib
   ```
2. Run the script:
   ```
   python app.py
   ```
3. The script will:
   - Download and preprocess the CIFAR-10 dataset
   - Train the CNN model for 10 epochs
   - Display sample images with their labels
   - Show accuracy plots during training
   - Save the trained model as `cnn_cifar10_model.h5`

## Sample Output
The script displays sample images from the dataset with their correct labels (as shown in Output.png) and plots the training progress, showing how accuracy improves over epochs.

## Future Improvements
- Experiment with deeper network architectures
- Try different regularization techniques (dropout, batch normalization)
- Implement data augmentation to improve generalization
- Fine-tune hyperparameters for better performance

## Dependencies
- TensorFlow 2.x
- Matplotlib
- NumPy
Training and validation accuracy graphs plotted using Matplotlib
![Screenshot (81)](https://github.com/user-attachments/assets/a4f18830-deae-4a68-8e2e-998d5308f971)

![Screenshot (83)](https://github.com/user-attachments/assets/d409b898-d9d2-40fb-a3b1-a0a02187dce8)
