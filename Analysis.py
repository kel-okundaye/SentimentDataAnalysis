# Import necessary libraries
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('movie_reviews')  # Movie reviews dataset
nltk.download('vader_lexicon')  # VADER sentiment analyzer lexicon
nltk.download('punkt')  # Tokenizer for breaking down sentences into words
nltk.download('stopwords')  # Common stopwords in English (e.g., 'the', 'and')
nltk.download('punkt_tab')  # This line might be unnecessary or redundant
from nltk.corpus import movie_reviews


# Function to load the movie reviews dataset
def load_movie_reviews():
    # Load positive reviews and label them 'pos'
    pos_reviews = [(movie_reviews.raw(fileid), 'pos') for fileid in movie_reviews.fileids('pos')]

    # Load negative reviews and label them 'neg'
    neg_reviews = [(movie_reviews.raw(fileid), 'neg') for fileid in movie_reviews.fileids('neg')]

    # Combine positive and negative reviews
    reviews = pos_reviews + neg_reviews

    # Create a pandas DataFrame with the reviews and their sentiments
    data = pd.DataFrame(reviews, columns=['Review', 'Sentiment'])
    return data


# Load the dataset into a DataFrame
data = load_movie_reviews()


# Function to preprocess the text (cleaning and normalizing the text)
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters, punctuation, and digits using regular expressions
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text into words
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # If tokenization fails (e.g., data not downloaded), fall back to basic splitting
        tokens = text.split()

    # Remove common English stopwords (e.g., 'the', 'is', etc.)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Join the cleaned tokens back into a single string
    return ' '.join(tokens)


# Apply text preprocessing to all the reviews in the dataset
data['Processed_Review'] = data['Review'].apply(preprocess_text)

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()


# Function to get sentiment using the VADER analyzer
def get_sentiment_vader(review):
    # Get sentiment scores for the review (e.g., positive, negative, neutral, compound)
    scores = sia.polarity_scores(review)

    # Classify sentiment as 'pos' (positive) if compound score is positive, otherwise 'neg'
    return 'pos' if scores['compound'] > 0 else 'neg'


# Apply the VADER sentiment analysis function to the preprocessed reviews
data['Predicted_Sentiment'] = data['Processed_Review'].apply(get_sentiment_vader)

# Create a confusion matrix comparing the actual sentiment to the predicted sentiment
cm = confusion_matrix(data['Sentiment'], data['Predicted_Sentiment'])

# Plot the confusion matrix using seaborn heatmap for visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['neg', 'pos'], yticklabels=['neg', 'pos'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Calculate the accuracy of the sentiment prediction
accuracy = accuracy_score(data['Sentiment'], data['Predicted_Sentiment'])
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plot the distribution of actual vs predicted sentiments
plt.figure(figsize=(12, 6))
sns.countplot(x='Sentiment', data=data, palette='coolwarm', label='Actual')  # Actual sentiments
sns.countplot(x='Predicted_Sentiment', data=data, palette='Set2', label='Predicted', alpha=0.7)  # Predicted sentiments
plt.legend()
plt.title('Distribution of Actual vs Predicted Sentiments')
plt.show()


# Function to analyze a custom review input by the user
def analyze_custom_review():
    # Prompt the user for input
    review = input("Enter a review: ")

    # Preprocess the custom review
    processed_review = preprocess_text(review)

    # Analyze the sentiment of the processed review using VADER
    sentiment = get_sentiment_vader(processed_review)

    # Print the predicted sentiment
    print(f'The predicted sentiment is: {sentiment}')


# Call the function to analyze a custom review input by the user
analyze_custom_review()
