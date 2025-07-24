import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Make sure NLTK resources are downloaded
nltk.download('stopwords')

# Load and clean dataset
@st.cache_data
def load_data():
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def clean_text(msg):
    msg = msg.lower()
    msg = ''.join([char for char in msg if char not in string.punctuation])
    msg = ' '.join([word for word in msg.split() if word not in stopwords.words('english')])
    return msg

# Load data and train model
df = load_data()
df['cleaned'] = df['message'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# App UI
st.title("📧 Email Spam Classifier")
st.write("Enter a message below to check if it's spam or not:")

user_input = st.text_area("Your Message", "")

if st.button("Detect"):
    clean_input = clean_text(user_input)
    input_vector = vectorizer.transform([clean_input])
    prediction = model.predict(input_vector)[0]
    st.success("🚫 It's a Spam Email!" if prediction == 1 else "✅ It's a Safe Email.")
