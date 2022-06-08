## Used Libraries
- numpy
- opencv
- skimage
- sklearn
- [Hinge and Cold Feature Extraction](https://github.com/Swati707/hinge_and_cold_feature_extraction)

## Pre-process the data
- read them as grayscale
- crop the part of the images with the text
- resize the images to a standard size
- convert the images to numpy arrays
- normalize the images /scale them to [0,1] `binary`
- split the data into train and test sets

## Extract the features
> extract the features from the cropped images using the following techniques
- [X] [COLD](./research/Cold.md)
- [X] [Hinge](./research/Hinge.md)
- [ ] LBP 
- [ ] HOG 
- [ ] GLCM, 
- [ ] Chain Code Based i.e.
    - [ ] Distribution of chain code
    - [ ] Distribution of chain code pairs
- [X] save the features in a file

## Train the model
> train the model using the extracted features best classifiers till now are 
- [X] SVM 
- [ ] ANN
- [X] save the model in a file

## Test the model
> test the model using the extracted features
- [X] SVM 
- [ ] ANN
- [X] save the results in a file

## Calculate the accuracy
- [X] calculate the accuracy of the model for each features set
- [ ] build a voting for each feature set to decide the final accuracy

## A Visualization Model 
- [ ] for the different features extractors
- [ ] for the different classifiers techniques


## Project Dirictory Structure
- D data-set 
  - D males  
  - D females
- D preprocessed 
  - D test
  - D data-set
    - D males
    - D females
- D features  (the code will generate this one)
  - F cold_features.npy
  - F hinge_features.npy
  - F labels.npz
- D classifiers
  - F svm_hinge_features_train.pkl
  - F svm_cold_features_train.pkl
- D test 
- D out
  - F results.txt  
  - F times.txt

#### Note: 
  - just mkdir each < D * > before running the code
  - the code will generate any < F * >



