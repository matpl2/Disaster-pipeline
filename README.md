Project 1 - Write A Data Science Blog Post
This file is a documentation file for an exercise performed as a part of Data Scientist NanoDegree project 1 - Write a Data Science blog.

Purpose of this project is to write a blog that will be supported by data exercise. In this file I will describe most useful information related to files I have uploaded to the repository. Files included are:

Project 1.ipnb - jupyter notebook.
Project 1.html - jupyter notebook exported to html file.
licence file.
data SEA.zip - zip file with data exported from Kaggle.
Purpose and intent
Dataset is available here: https://www.kaggle.com/airbnb/seattle/data. Purpose of this excercise is to find answer to 3 questions (provide 3 insights):

Question 1 - Does cleaning fee impact cleanliness that was assessed by visitors in the review scores cleanliness rank?

Question 2 - Does the host tenure on AirBnB impact the accuracy of property description?

Question 3 - Which areas of Seattle get the best scores on the location?

Code
Data analysis was executed in a Jupyter notebook that operates on Python 3 Kernel (Anaconda). Python libraries used for this project includes:

numpy - https://numpy.org/
pandas - https://pandas.pydata.org/
pyplot - https://matplotlib.org/api/pyplot_api.html
sklearn modules: linelar_model, r2_score, model_selection - https://scikit-learn.org/stable/
seaborn - https://seaborn.pydata.org/
For the purpose of this excercise I have created two functions: removecharacter and movetodtype. Specification of these functions are:

removecharacter - This function removes string [a] in colum names [cols] provided as a list for a given panda dataframe [df]. cols must be provided as a list in form ['a','b','c'].

movetodtype - This function convert to given format [form] any colum names [cols] provided as a list for a given panda dataframe [df]. cols must be provided as a list in form ['a','b','c'] while form needs to be valid pandas data type like 'str' or 'float64'.

Analytics Process
Process outlied in the Project file follows CRISP-DM Process (Cross Industry Process for Data Mining). You can see inside the file that multiple steps were taken to prepare data: drop nan values, change data types, or remove special characters from the data. File was documented with comments therefore each step has instruction associated with it.

Outcome of the analysis
Analysis has provded answer to 3 business questions. You are encuraged to dive more into the dataset. More details can be found in my post:

https://medium.com/@matploskonka/airbnb-and-seattle-what-can-you-get-from-this-match-a6c813baa3c7
