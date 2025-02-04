This project builds a machine learning model to predict whether a customer will respond to a marketing campaign. The dataset contains customer demographics, purchasing behavior, and campaign interactions.

Features Used
•	Total Spending: Sum of all product category spending
•	Total Purchases: Sum of all purchase channels
•	Children: Sum of kids and teens in the household
•	Encoded categorical features (Education, Marital Status)

Model & Performance
•	Model Used: Random Forest Classifier
•	Accuracy: 87.61%
•	Class 0 (No Response): Precision = 89%, Recall = 97%
•	Class 1 (Response): Precision = 62%, Recall = 29%

The model performs well in predicting non-responders but struggles with accurately identifying responders due to class imbalance. Addressing this issue and improving feature engineering could enhance prediction accuracy.
