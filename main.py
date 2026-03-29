import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

data = {
    'text': [
        'I love this product, it is amazing!',
        'Horrible experience, not recommended.',
        'It is okay, nothing special.',
        'Absolutely fantastic, exceeded my expectations.',
        'Worst purchase ever, very disappointed.',
        'Neutral feelings about this item.',
        'Great value and quality for the price.',
        'Not bad, but could be better.',
        'Terrible customer service!',
        'Satisfied with the product overall.'
    ],
    'sentiment': [
        'positive', 'negative', 'neutral', 'positive', 'negative', 
        'neutral', 'positive', 'neutral', 'negative', 'positive'
    ]
}

df = pd.DataFrame(data)


X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
