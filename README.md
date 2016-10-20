# Tensorflow Document Classification
---
A Tensorflow model to classify documents using Multinomial Logistic Regression (softmax regression). 

#### Stuff done:
- [x] Clean the data (remove numbers, special characters, stops words)
- [x] Build the testing and training feature matrix using percentage split
- [x] Normalize the dataset
- [x] Build train and test csv files
- [x] Apply TFIDF
- [x] Apply Softmax Regression
- [x] Build a bigger list of commonly used words

#### To-do:
- [ ] Add more files for training as well as prediction
- [ ] k fold
- [ ] overfitting prevention (dropout(https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html))
- [ ] sparse matrix
- [ ] find optimal epoch
- [ ] find optimal learning rate
- [ ] check for new error functions
- [ ] check softmax_cross_entropy_with_logits
- [ ] Write a python script to automate the entire process
- [ ] Fix matplotlib window bug
