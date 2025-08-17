# Applications Deep Learning Robotics
_Repository for my undergraduate thesis: Deep Learning in Mechatronic Engineering. Includes code and resources for image classification, image segmentation, and reinforcement learning._

## About
This repository contains the various applications developed in my mechatronic engineering thesis, focused on Artificial Intelligence, specifically Machine Learning / Deep Learning.
The thesis is titled: “Theoretical-Practical Manual of Deep Learning Applied to Mechatronic Engineering Problems.”
Different machine learning techniques are applied, such as:

* Image Classification
* Reinforcement Learning
* Image Segmentation

Additionally, several theoretical concepts of machine learning are explained, including:

* Neural Networks
* Convolutional Neural Networks (CNNs)
* CNN Architectures
* Transfer Learning

## Technologies
The projects are implemented in Python using several libraries:
* Keras
* TensorFlow
* OpenCV (CV2)
* numpy
* Stable_baseline3

## Usage
To use each application, the corresponding imported packages must be installed.
### ImagenClasification_TransferLearning
* TensorFlow
* Pandas
* matplotlib
* Numpy
* seaborn
  
### Motion_control-Reinforcement_learning
* gym
* torch
* stable_baseline3
* pybullet
  
### Environment_Identification-Image_Segmentation
* Numpy
* CV2
* frozen_inference_graph_coco.pb
* mask_rcnn_inception_v2_coco_2018_01_28.pbtxt

The last two files are pre-trained COCO models available [here](https://github.com/methylDragon/opencv-python-reference/tree/master/Resources/Models/mask-rcnn-coco)

## Resumen
In recent years, Artificial Intelligence (AI) has experienced remarkable growth, showing advances in various sectors and transforming daily life, tasks, and industries.

One field particularly impacted is robotics, with notable progress in collaborative robotics, process automation, and autonomous vehicles.

Mechatronic engineering is a discipline highly representative of automation and robotics; however, until now, it has had a limited focus on AI, which restricts the academic and professional opportunities that AI mastery could offer.

To address this gap, this project develops a theoretical-practical manual featuring three AI-powered applications, extremely useful for robot implementation and mechatronic engineering.

The objective is to allow any professional interested in the field to acquire the necessary knowledge to work effectively in AI-based robotics.

The applications include:

* Image Classification
* Semantic Image Segmentation for object detection in environments
* Reinforcement Learning for robot motion development

Application Overview
### Image Classification – Transfer Learning
Created an image classifier trained with a fruit dataset [available here](https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification).

Where a fruit image is input, and the result is a one-hot encoded classification among the 5 possible classes:

<img src="ImagenClasification_TransferLearning/Clasificacionfrutas.PNG" align="center" alt="Clasificacion de frutas" width="500">

In this example, 5 fruits from each class were input, and the model correctly returned the corresponding class for each.

<img src="ImagenClasification_TransferLearning/PrediccionFrutas.png" align="center" alt="Prediccion frutas" width="500">

Finally, predictions were made with a test dataset to verify the results of each model, and a confusion matrix was obtained for each. Here is the one for the ResNet50 model:

<img src="ImagenClasification_TransferLearning/MatrizConfusionResNet50.png" align="center" alt="Matriz de confusion del modelo RESNET50" width="500">

This shows that the identity line contains the majority of predictions, indicating correct model training.


### Motion Control – Reinforcement Learning
In this application, a robot was trained in a virtual environment where the agent (ant robot) had no instructions on how to move and was rewarded when it performed a movement that changed its position in the environment.

<img src="Motion_control-Reinforcement_learning/RLAntrain.PNG" align="center" alt="Agente en su simulacion" width="500">

After many iterations and hyperparameter adjustments, the robot developed a movement method to navigate efficiently. The following image shows the learning progress and the "score" achieved by the robot at each iteration, demonstrating how the robot learned:

<img src="Motion_control-Reinforcement_learning/GraficotrainRL1.PNG" align="center" alt="Grafico de entrenamiento" width="500">

### Environment Identification – Image Segmentation
In this application, the environment components were identified using the mask-RCNN Inception model, trained with the COCO dataset. In the image below, the model identifies different objects from an input image and assigns each a category with a mask color:

<img src="Environment_Identification-Image_Segmentation/Prediction2rcnn.PNG" align="center" alt="Prediccion" width="500">

The implementation was done using CV2 and the pre-trained model. Here, you can see the segmentation on an image of the Universidad Militar Nueva Granada:

<img src="Environment_Identification-Image_Segmentation/SegmentacionTotal.png" align="center" alt="Prediccion UMNG" width="500">

## Document 📖

The full undergraduate thesis document is available here: [UMNG](https://repository.unimilitar.edu.co/items/2895e290-f05f-44e9-9e1a-34f3fc5c3e97)

## Tutor ✒️

This project was conducted under the supervision of:
* **Nelson Fernando Velasco Toledo** - Universidad Militar Nueva Granada [GitHub](https://github.com/nelsonfvt) [Linkedin](https://www.linkedin.com/in/nelson-fernando-velasco-toledo-1a6b6b249/)

[MaickMos](https://github.com/Maickmos)
