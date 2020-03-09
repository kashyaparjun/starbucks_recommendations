# Starbucks recommendation strategy

## Project overview
This is the time and age where customer expect personalized experience no matter how big the size of company is. A huge company like Starbucks with millions of users is expected to customize the experience for every user. In such scenarios Data Science can be used to learn from customer behaviour and patterns to suggest customers with what they would potentially like. The dataset used for this project is a simulated set of actual customer behaviour at Starbucks. Starbucks usually provides offers to its customers such as Buy One Get One, discounts, and product advertisements. 

These informations need to be targetted so that the required customer buys the product. To do so, some preliminary analysis and predictions are required. For the current use case, I have used the customer age, gender, income and the date of membership to fuel this decision.

The prediction from this model would be strategy that would contain, what type of recommendation, duration of offer, the difficulty of conversion for this user and aslo the communication mediums for the recommendation are predicted. Using these, a recommendation can be made so that the user buys one of Starbucks products.

The CRISP-DM process is followed during this project:
1. Business understanding
2. Data Understanding
3. Data Preparation
4. Modelling
5. Evaluation
6. Deployment

## Assessments and metrics
The train-test split was 80-20. That is, 80% of the dataset was used for training while the 20% of it was used for testing. Accuracy, precision, recall and f1 scores were calculated to mesure the models performance. RandomForestClassifier with GridSearchCV was used to fit and predict the model. The estimator sizes were: 10, 50 and 100.

## Packages and softwares used
1. Python 3.7+
2. pandas
3. numpy
4. matplotlib
5. seaborn
6. sklearn
7. plotly
8. flask

All these can be installed using the anaconda python distribution

## File Description
The pickle(.pkl) files are in the google drive link: `https://drive.google.com/open?id=1WxoR21Hh9qVnkrpxf3fQz2ceNlJ6mu9c` this is due to the git limit 100 MB per file. Please download these and unzip before you procee.
1. data/* - these contain the Starbucks datasets. More on these can be found in the jupyter notebook
2. Starbucks_Capstone_notebook.ipynb - contains the code and output of model building, observation, inference, etc.
3. *_model.pkl - These are the stored model for prediction
4. filtered_data.pkl - This is the pickle file containing filtered data that is used to display data sample in the web app
5. app/run.py - this is the main file containg logic for the webapp

## Instructions to run the webapp
1. Make sure that all the required packages and softwares are installed
2. `cd` into the app folder
3. run the code `python run.py`
4. This will start a server.
5. Open the url `http://0.0.0.0:3001` in the browser
6. Input sample data and test