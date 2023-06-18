# Movie Review Sentiment Analysis and Text Classification
This repository combines two projects: Sentiment Analysis and Text Classification. It aims to provide a comprehensive analysis of movie reviews using natural language processing techniques. The goal is to predict the sentiment (positive or negative) of movie reviews and classify them into different categories based on their text content.

# Project Overview
The project consists of two main components:

1. __Sentiment Analysis__:

- __Dataset__: The project uses a movie review dataset containing labeled reviews with their respective sentiments.
- __Approach__: The sentiment analysis component utilizes the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool. The SentimentIntensityAnalyzer from the NLTK library is used to assign sentiment scores to each review. A compound score is computed, representing the overall sentiment. Based on the compound score, a predicted label ("pos" or "neg") is assigned to each review.
- __Evaluation__: The accuracy of the sentiment analysis model is calculated by comparing the predicted labels with the actual labels in the dataset. Classification metrics such as precision, recall, and f1-score are calculated, and a confusion matrix is generated to evaluate the model's performance.

2. __Text Classification__:

- __Dataset__: The project uses a movie review dataset with labeled reviews and corresponding categories.
- __Approach__: The text classification component utilizes machine learning techniques to classify movie reviews into different categories. It involves preprocessing the text data, extracting features using techniques like TF-IDF (Term - Frequency-Inverse Document Frequency), and training a classification model such as Naive Bayes, Support Vector Machines (SVM), or Random Forest.
- __Evaluation__: The trained classification model is evaluated using metrics like accuracy, precision, recall, and F1-score. The performance of different models and feature extraction techniques can be compared to determine the most effective approach.
- __Repository Structure__
The repository contains the following files and folders:

- ├── sentiment_analysis/
- │   ├── sentiment_analysis.ipynb
- │   └── moviereviews.tsv
- ├── text_classification/
- │   ├── text_classification.ipynb
- │   └── moviereviews_category.tsv
- └── README.md

1. __sentiment_analysis/__: This folder contains the files related to the sentiment analysis component.
- sentiment_analysis.ipynb: Jupyter Notebook file that implements the sentiment analysis component. It includes code and explanations for each step of the sentiment analysis process.
- moviereviews.tsv: Dataset file containing movie reviews with labeled sentiments.

2. __text_classification/__: This folder contains the files related to the text classification component.
- text_classification.ipynb: Jupyter Notebook file that implements the text classification component. It includes code and explanations for preprocessing, feature extraction, model training, and evaluation.
- moviereviews.tsv: Dataset file containing movie reviews with labeled categories.

# Technologies Used
The project incorporates the following technologies:
1. Python: The programming language used for implementing the sentiment analysis and text classification components.
2. Natural Language Processing (NLP) libraries: Utilized NLTK library for sentiment analysis and various NLP preprocessing tasks.
3. scikit-learn: Employed scikit-learn library for text preprocessing, feature extraction, and classification models.
4. Jupyter Notebook: Used to provide an interactive and user-friendly environment for running and modifying the code.

# Usage
To use this movie review analysis system:
1. Clone the repository to your local machine.
2. Install the required dependencies, including Python, NLTK, scikit-learn, and Jupyter Notebook.
3. Explore the sentiment_analysis/ and text_classification/ folders to access the respective Jupyter Notebook files.
4. Open the desired notebook (sentiment_analysis.ipynb or text_classification.ipynb) and run the cells sequentially to understand the code and reproduce the analysis.
5. Feel free to modify the code and experiment with different approaches, models, or datasets.
6. Refer to the documentation within the notebook files for detailed explanations and instructions.
   
# Credits
This project was inspired by the "NLP - Natural Language Processing with Python" course on Udemy, taught by Jose Portilla. The code and concepts from the course were utilized to implement the sentiment analysis and text classification components.
