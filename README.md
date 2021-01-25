# Building Models to Predict Pedestrian Traffic Threshold
This program uses Pyspark to process big data and performs binary target prediction using Decision Tree and Gradient Boost Tree Algorithms from MLlib.

* **Pedestrian_Counting_System_-_Monthly__counts_per_hour.csv**: https://drive.google.com/file/d/12PE127tr-OeKY487gjXOYSnMesHDOAxl/view?usp=sharing

* **Pedestrian_Counting_System_-_Sensor_Locations.csv**: The dataset with information regarding each sensor, including the geographical coordinates, names and direction of the sensors. 

* **PySpark_BigData_ML_BinaryTargetWithDT_GBT.ipynb**: Jupyter notebook processing and training big data. The notebook includes the following sections for building and training models: 

  1. **Loading Data with Pre-defined Schema**
  2. **Exploratory Data Analaysis** - exploring individual features, including statiscal characteristics, correlations of each and every numerical feature, as well as the distribution of numerical variables 
  3. **Visualisation of Target variable vs. Features** - visualising the change in ``Hourly_Counts`` attribute based on different features 
  4. **Feature Extraction** - Deciding the features to use for the model and make adjustment to the dataset accordingly; includes extensive use of visualisation using ``matplotlib`` library 
  5. **Prepare for Train/Test dataset and Transform Features** - using ``ml`` library from ``pyspark``, transform categorical variables with ``StringIndexer`` and ``OneHotEncoder``. Also transform all the numerical features and binary vectors from ``OneHotEncoder`` using ``VectorAssembler``
  6. **Pipelining Transformation and Model Fitting Process** - initialise Decision Tree and Gradient Boost Tree Classifier and built pipelines for all the transformation process at step 5. 
  7. **Fit the Data and Evaluate the Model**: After fitting the test dataset into the models, evaluate the performance of the models using confusion matrix and AUC-ROC. 
  8. **Fine-Tune Hyperparameters and Select the Best Model Using CrossValidation**
  9. **Save the Model to the File System**

