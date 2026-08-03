## Project Report ##
Tinea and Candidiasis Detection - Group E13

## Dataset ##
We used the [Skin Diseases](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset) from Kaggle to get the **Tinea** and **Candidiasis** dataset, containing a total of 1109 images for the two dataset combined.

## Application ##
Our streamlit application accepts uploaded images of either Tinea or Candidiasis and clasifies it using a fine-tuned MobileNetV2 model. The model achieve 89.65% accuracy.

## Challenges ##
Our main challenge was training compute - we relied on Kaggle's free-tier GPU, which limited session time and hyperparameter exploration. Deploying to Streamlit Community Cloud required pinning Python 3.12, as TensorFlow does not support the latest version of Python yet. We also had to use tensorflow-cpu to stay within the platform's 1 GB memory limit.

## Possible Improvements ##
With more compute, we would explore larger backbones and add Grad-CAM visualisation to the app so inspectors can see which regions the model flagged. We will also train it to identify more images of different skin diseases.