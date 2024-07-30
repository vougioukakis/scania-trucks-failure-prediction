# scania-trucks-failure-prediction
Machine Learning project to predict failures in air pressure system of Scania trucks, and minimize cost for the manufacturer.
Any features that had more than 35,000 missing values were removed to ensure the quality of our data.
The best model was the Random Forest with class weights, using Random oversampling, and per class median imputation, with scores ranging from 5500 to 7000.
