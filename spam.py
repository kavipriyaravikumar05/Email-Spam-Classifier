import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
# Download stopwords (only first time)
nltk.download('stopwords')
from nltk.corpus import stopwords
# 1. Load Dataset
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
# Convert labels: ham = 0, spam = 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
# 2. Text Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text
data['message'] = data['message'].apply(clean_text)
# 3. Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(data['message'])
y = data['label']
# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 5. Train Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)
# 6. Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# 7. Predict New Message
def predict_message(message):
    message = clean_text(message)
    vector = vectorizer.transform([message])
    prediction = model.predict(vector)
    return "SPAM" if prediction[0] == 1 else "HAM"
# 8. Test Examples
print("\nTest Results:")
print("Message: Congratulations! You won a free prize")
print("Prediction:", predict_message("Congratulations! You won a free prize"))
print("\nMessage: Are we meeting today at 5 PM?")
print("Prediction:", predict_message("Are we meeting today at 5 PM"))