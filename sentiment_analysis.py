import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Download required NLTK data (only first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Step 1: Load Data ---
# CSV file should have columns: review_id, review_text, rating (1-5 stars)
data = pd.read_csv('product_reviews.csv')

# Map ratings to sentiment
def map_rating_to_sentiment(rating):
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:
        return 'positive'

data['sentiment'] = data['rating'].apply(map_rating_to_sentiment)

# Check distribution of sentiments
print("Sentiment distribution:")
print(data['sentiment'].value_counts())

# --- Step 2: Text Preprocessing ---

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    # Join back to string
    return ' '.join(filtered_tokens)

# Apply preprocessing
data['cleaned_text'] = data['review_text'].apply(preprocess_text)

# --- Step 3: Exploratory Data Analysis (EDA) ---

# Plot sentiment distribution
plt.figure(figsize=(6,4))
data['sentiment'].value_counts().plot(kind='bar', color=['red','gray','green'])
plt.title('Sentiment Class Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.show()

# WordCloud for all reviews combined
all_text = ' '.join(data['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of All Reviews')
plt.show()

# --- Step 4: Feature Extraction ---
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['sentiment']

# --- Step 5: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Step 6: Model Training and Evaluation ---

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

for model_name, model in models.items():
    print(f"\nTraining and evaluating model: {model_name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    with open(f"{model_name.replace(' ', '_').lower()}_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    print(f"{model_name} model saved.")

# --- Step 7: Save TF-IDF Vectorizer ---
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("TF-IDF Vectorizer saved.")

