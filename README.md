# Project Name: Streaming Service Classifier (Web APIs & NLP)
# Done by: Lua Yeow Wen DSIF 11
## Content:
### 1. Overview
### 2. Problem Statement
### 3. Data Collection
### 4. Data Cleaning
### 5. Modelling
### 6. Model Comparison
### 7. Conclusion
### 8. Recommendations
### 9. Further Improvements


## Jupyter Notebooks
* Data Collection: 01_data_collection
* Data cleaning and EDA: 02_data_clean_eda
* Modelling: 03_data_modelling

## Version
* The version of the Jupyter notebook server is: 6.5.4
* Python 3.11.3 | packaged by Anaconda, Inc. | (main, Apr 19 2023, 23:46:34) [MSC v.1916 64 bit (AMD64)]


## 1. Overview

Reddit serves as a social media platform hosting a myriad of communities, allowing individuals to explore their interests, hobbies, and passions. These communities, referred to as Subreddits, are dedicated to specific topics. The objective of this project is to develop a machine learning model capable of topic classification, focusing on insights related to the chosen topics (Netflix and Disneyplus).



## 2. Problem Statement

Netflix is facing threat in their market share of subscribers to Disneyplus as their subscriber numbers has been surpassed. With the additional joint ownership of Hulu and ESPN by Disneyplus, it added more disadvantage to Netflix in the show streaming industry. Moreover, Netflix also lost 18 billion dollars in value that cause shareholders to be unhappy with the company. The main source of revenue for Netflix comes from the subscribers and this source of revenue has been threathened.


sources:
* https://www.forbes.com/sites/qai/2022/09/27/disney-surpasses-netflix-subscriber-count-what-does-that-means-for-investors/?sh=66739a245e0b
* https://www.gamingbible.com/news/tv-and-film/netflix-just-lost-18-billion-in-value-949478-20230721

### Scope
Create a classification model for Netflix and provide key insights to ensure their marketing campaigns and shows aligns with keywords and themes closely associated with Disneyplus.


### Goal
* Achieve a minimum of 90 percent F1 score.
* Strategically redirect online search traffic towards Netflix from Disneyplus with keywords and themes closely associated with Disneyplus on Reddit. 


### Why are we doing this:
* This strategic maneuver will help Netflix solidify its position in the market and attract more viewers to their platform.

### Who we are:
A group of data science consultant engaged by the Netflix marketing team.


### Primary Stakeholders: 
* Netflix Marketing Team

### Secondary Stakeholders: 
* Netflix Content Team


### Data Sources

The data were scraped from Reddit by using Praw (Python Reddit API Wrapper) to access Reddit, tabulate and save its data as CSV with Pandas.

Praw (Python Reddit API Wrapper)
* https://www.reddit.com/r/Netflix
* https://www.reddit.com/r/Disneyplus


### Data Files for Data Cleaning and EDA
* netflix_df: Netflix_reddit_submissions.csv
* disneyplus_df: Disneyplus_reddit_submissions.csv



## 3. Data Collection

The data were scrapped from 5 categories from each subreddits:
* hot
* new
* rising
* controversial
* top

The features that will be extracted from each categories are:

* subreddit: topic_num,
* title: submission.title,
* selftext: submission.selftext,
* ups: submission.ups,
* upvote_ratio: submission.upvote_ratio,
* num_comments: submission.num_comments,
* author: str(submission.author),
* link_flair_text: submission.link_flair_text,
* awards: len(submission.all_awardings),
* is_original_content: submission.is_original_content,
* is_video: submission.is_video,
* post_type: 'text' if submission.is_self else 'link',
* domain: submission.domain,
* created_utc: submission.created_utc,
* pinned: submission.pinned,
* locked: submission.locked,
* stickied: submission.stickied 
  
Thereafter, the dataset of each topic will be saved as the below following before Data Cleaning and EDA:
* Netflix data: Netflix_reddit_submissions.csv
* Disneyplus data: Disneyplus_reddit_submissions.csv



## 4. Modelling

### Preprocess the text

Lemmatization is chosen over stemming because it analyzes the context of a word, transforming it into its essential and meaningful base form. Despite the typically longer processing time compared to stemming, choosing stemming can negatively impact model performance due to its tendency to produce inaccurate meanings and spellings.

POS tagging is also included during Lemmatization because it helps computers understand the structure and meaning of words in sentences.

Removal of Stopwords will help to speed up the analysis as noise is reduced. These process will also hasten and enable us to focus on the important words.

* Pos Tagging
* Lemmatization
* Removal of Stopwords





## 5. Modelling

A total of 8 models with CountVectorizer with Multinomial Naive Bayes as Baseline will be executed and identify the best model to classify.
<br>
<br>
**CountVectorizer with Multinomial Naive Bayes (Baseline)** is chosen as the baseline model as I want to see how well a model that does not take the context of a sentence when used with others words performs (Text data is never independent). Moreover, it is also a very fast modeling algorithm.
<br>
<br>
From here, other longer processing time models will also be included for comparisons
<br>
<br>
Various hyperparameters will also be included and the help of GridSearch CV will provide the best results based on the different combinations of hyperparameter values each model.




### Machine Learning Models


#### CountVectorizer

* Model 1 - CountVectorizer with Multinomial Naive Bayes (Baseline)
* Model 2 - CountVectorizer with KNNeighbours
* Model 3 - CountVectorizer with Logistic Regression
* Model 4 - CountVectorizer with Random Forest


#### TfidfVectorizer

* Model 5 - TfidfVectorizer with Multinomial Naive Bayes
* Model 6 - TfidfVectorizer with KNNeighbours
* Model 7 - TfidfVectorizer with Logistic Regression
* Model 8 - TfidfVectorizer with Random Forest

### Evaluation Metric
F1 Score will be used as the metric to evaluate the best performing model as it is crucial for the model to demostrate its effectiveness in distinguishing between the two classes (1 for Netflix and 0 for Disneyplus).

1. Minimising False Positives: A high F1 score means our model excels at correctly identifying text as belonging to Class 0 (Disneyplus) without mistakenly categorising it as Class 1 (Netflix) posts. This ensures that our model avoids making false positive predictions.

2. Minimising False Negatives: A high F1 score ensures that our models does not overlook relevant text belonging to Class 0 (Disneyplus). It effectively captures and classifies such text, avoiding false negatives.




## 6. Model Comparison


| **Vectorizer** | **Model**      | **Best Score** | **Train Score** | **Test Score** | **Train Accuracy** | **Test Accuracy** | **Train F1 Score** | **Test F1 Score** | **Train ROC-AUC** | **Test ROC-AUC** | **Execution Time  (Seconds)** | **Remark** |
|----------------|----------------|----------------|-----------------|----------------|--------------------|-------------------|--------------------|-------------------|-------------------|------------------|-------------------------------|------------|
| CVEC           | Multinomial NB | 0.898          | 0.927           | 0.908          | 0.927              | 0.908             | 0.926              | 0.906             | 0.982             | 0.972            | 263.7                         | Baseline   |
| CVEC           | KNN            | 0.864          | 0.997           | 0.859          | 0.997              | 0.859             | 0.997              | 0.835             | 0.997             | 0.854            | 1414.31                       | -          |
| CVEC           | Log Reg        | 0.927          | 0.988           | 0.924          | 0.988              | 0.924             | 0.988              | 0.916             | 0.999             | 0.984            | 551.23                        | -          |
| CVEC           | Random Forest  | 0.900          | 0.913           | 0.888          | 0.913              | 0.888             | 0.900              | 0.870             | 0.984             | 0.971            | 1873.82                       | -          |
| TVEC           | Multinomial NB | 0.906          | 0.944           | 0.912          | 0.944              | 0.912             | 0.942              | 0.909             | 0.990             | 0.977            | 253.96                        | -          |
| TVEC           | KNN            | 0.772          | 0.875           | 0.766          | 0.875              | 0.766             | 0.849              | 0.694             | 0.868             | 0.755            | 1823.63                       | -          |
| **TVEC**       | **Log Reg**    | 0.934          | **0.993**       | 0.935          | 0.993              | **0.935**         | 0.993              | **0.931**         | 0.999             | **0.984**        | 592.46                        | -          |
| TVEC           | Random Forest  | 0.903          | 0.921           | 0.887          | 0.921              | 0.887             | 0.909              | 0.871             | 0.986             | 0.969            | 1996.41                       | -          |



Observations from the table above:

Amongst all the model, CountVectorizer and TfidfVectorizer with both KNNeighbours and Random Forest did not perform well in our metric (F1 Score). They also consume the most processing time. 
<br/>

Even though all the models are overfitted due to data imbalance, Multinomial Naive Bayes and Logistic Regression are still able to meet the threshold of 10%.
<br/>

TfidfVectorizer with Logistic Regression attained the best F1 Score of 0.931 with execution time of 592.46s.
It is one of the fastest running model which is crucial as there will be more data to be train and tested which will lead to longer processing time.
<br/>

TfidfVectorizer with Logistic Regression also performs well with the mixed of unigrams and bigrams.

The ROC AUC score is close to 1, implying that this model is good at distinguishing between Netflix and Disneyplus topics as higher true positive rate and a lower false positive rate indicates a better balance between sensitivity and specificity.


## 7. Conclusion
Based on the finding in model comparison, we will propose our TfidfVectorizer with Logistic Regression model to our stakeholders (Netflix marketing team).

The model is able to distinguishing between Netflix and Disneyplus topics so thatthe words that are purposely picked from Disneyplus for Netflix marketing campaign will stand higher chance for their posts to be seen by Disneyplus subscribers on reddit as they key in the words in the search engine.

With short processing time, we can constantly provide different keywords to our stakeholders with a short turnaround time. This is so that the marketing team can quickly have the content team to work on newer contents with words that are impactful in the show names.


Keywords that are valuable for classifying content related to Disneyplus include mandalorian, marvel, wandavision,loki, hulk can also help the Netflix team to understand the content that Disneyplus prefer and create shows that with similar genres.

The Netflix marketing can also use some of the generic words such as poster, tangle, entry, born and frozen (although it is a show name) to make up words for marketing campaign or content creation, drifting disneyplus subscribers to netflix.

However, there are some limitations with our model such as:
* The content and associated keywords for both platforms are constantly evolving.
* Using Disneyplus keywords on Netflix and retraining the model may blur semantic distinctions between the two platforms over time.
* Text classification models can identify keywords but might struggle with understanding context or sentiment.



## 8. Reccomendations
* Leverage on Popular Disney Themes: 
Develop content in trending genres like sci-fi and superheroes to cater to DisneyPlus fans seeking variety.

* Enhance User Experience: 
Prioritize platform usability and innovative features to outshine competitors and retain customers.

* Content Marketing: 
Spotlight unique Netflix content to position the platform as either a complement or superior alternative to DisneyPlus.


## 9. Further Improvements
* Data Expansion: Collect more data over time to improve the model's robustness
* Data Imbalance: Implement algorithm such as SMOTE to handle increasing data imbalance over time.
* Sentiment Analysis: Incorporate sentiment scores as features to understand the sentiment behind each text.
* Model Exploration: Try more advanced models like deep learning for text classification.
* Ensemble Methods: Combine predictions from multiple models to improve metrics.
