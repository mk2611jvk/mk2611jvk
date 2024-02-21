# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# Load the data
# Assuming the data has columns: salesman_id, salesman_name, customer_id, customer_name, review, rating, photo
data = pd.read_csv("salesman_reviews.csv")

# Explore the data
# Print the first 5 rows
print(data.head())

# Print the summary statistics
print(data.describe())

# Plot the distribution of ratings
sns.countplot(data=data, x="rating")
plt.title("Distribution of Ratings")
plt.show()

# Analyze the sentiment of reviews
# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to get the sentiment score
def get_sentiment_score(review):
    # Get the polarity scores
    scores = sid.polarity_scores(review)
    # Return the compound score
    return scores["compound"]

# Apply the function to the review column
data["sentiment_score"] = data["review"].apply(get_sentiment_score)

# Plot the distribution of sentiment scores
sns.histplot(data=data, x="sentiment_score")
plt.title("Distribution of Sentiment Scores")
plt.show()

# Generate a word cloud of positive reviews
# Filter the data for positive reviews
positive_reviews = data[data["sentiment_score"] > 0.5]

# Join all the positive reviews into one text
positive_text = " ".join(positive_reviews["review"])

# Generate and display the word cloud
positive_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(positive_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Positive Reviews")
plt.show()

# Generate a word cloud of negative reviews
# Filter the data for negative reviews
negative_reviews = data[data["sentiment_score"] < -0.5]

# Join all the negative reviews into one text
negative_text = " ".join(negative_reviews["review"])

# Generate and display the word cloud
negative_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(negative_text)
plt.figure(figsize=(10, 5))
plt.imshow(negative_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Negative Reviews")
plt.show()

# Display some sample reviews and photos
# Define a function to display a review and a photo
def display_review_and_photo(index):
    # Get the row of data
    row = data.iloc[index]
    # Print the review and the rating
    print(f"Review: {row['review']}")
    print(f"Rating: {row['rating']}")
    # Display the photo
    plt.imshow(plt.imread(row["photo"]))
    plt.axis("off")
    plt.show()

# Display some random reviews and photos
np.random.seed(42) # For reproducibility
random_indices = np.random.randint(0, len(data), 5) # Get 5 random indices
for i in random_indices:
    display_review_and_photo(i)
