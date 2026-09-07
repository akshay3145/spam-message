import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Instantiate Lemmatizer and Stopwords globally or locally
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans the input text by performing:
    - Lowercasing
    - Punctuation removal
    - Tokenization
    - Stopword removal
    - Lemmatization
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Punctuation removal
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Tokenization
    # Using nltk word_tokenize instead of simple split()
    # If punkt is missing this will fail, hence the download check above.
    # We can also use a fallback but let's stick to the pipeline requested.
    # Fallback to simple split if word_tokenize fails somehow:
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download('punkt_tab')
        tokens = word_tokenize(text)
    
    # 4 & 5. Stopword removal and lemmatization
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)

def load_and_preprocess_data(file_path):
    """
    Loads data and applies preprocessing.
    """
    print("Loading data...")
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        print("Using dummy data for demonstration purposes...")
        df = pd.DataFrame({
            'Category': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
            'Message': [
                'Hi, I will be late for the meeting.',
                'WINNER!! You have won a free lottery ticket. Call 123456 now!',
                'Can you pick up groceries on your way back?',
                'Urgent! Your bank account needs verification. Click link to secure.',
                'Let\'s catch up over the weekend.',
                'Get cheap loans instantly. No credit check required. Apply now.',
                'Sure, I will send the report by tonight.'
            ]
        })

    # Encode labels (spam = 1, ham = 0)
    df['Label'] = df['Category'].map({'spam': 1, 'ham': 0})
    
    print("Applying text cleaning (this might take a few moments for a large dataset)...")
    df['Cleaned_Message'] = df['Message'].apply(clean_text)
    
    return df

def train_and_evaluate_model(df):
    """
    Extracts features, trains the model, and evaluates it.
    """
    print("Extracting features using TfidfVectorizer...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Cleaned_Message'])
    y = df['Label']
    
    # Split the dataset: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    # Prevent warnings if target_names are missing in small dummy dataset
    labels_present = sorted(list(y_test.unique()))
    target_names = ['Ham', 'Spam']
    target_names_present = [target_names[i] for i in labels_present]
    
    print(classification_report(y_test, y_pred, target_names=target_names_present, labels=labels_present))
    
    return model, vectorizer

def visualize_data(df):
    """
    Generates word clouds and frequency bar charts.
    """
    print("Generating visualizations...")
    
    # Separate data
    spam_messages = " ".join(df[df['Label'] == 1]['Cleaned_Message'])
    ham_messages = " ".join(df[df['Label'] == 0]['Cleaned_Message'])
    
    plt.figure(figsize=(15, 6))
    
    # Spam Word Cloud
    if spam_messages.strip():
        spam_wc = WordCloud(width=600, height=400, background_color='black', colormap='Reds').generate(spam_messages)
        plt.subplot(1, 2, 1)
        plt.imshow(spam_wc, interpolation='bilinear')
        plt.title('Spam Messages Word Cloud')
        plt.axis('off')
    
    # Ham Word Cloud
    if ham_messages.strip():
        ham_wc = WordCloud(width=600, height=400, background_color='white', colormap='Greens').generate(ham_messages)
        plt.subplot(1, 2, 2)
        plt.imshow(ham_wc, interpolation='bilinear')
        plt.title('Ham Messages Word Cloud')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig('wordclouds.png')
    print("Word clouds saved as 'wordclouds.png'.")
    
    # Top 10 Frequent Words
    all_words = " ".join(df['Cleaned_Message']).split()
    word_counts = Counter(all_words)
    top_10_words = dict(word_counts.most_common(10))
    
    plt.figure(figsize=(10, 6))
    plt.bar(top_10_words.keys(), top_10_words.values(), color='skyblue', edgecolor='black')
    plt.title('Top 10 Most Frequent Words in Dataset')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.savefig('top_words.png')
    print("Top words bar chart saved as 'top_words.png'.")

def predict_custom_sms(message, model, vectorizer):
    """
    Predicts whether a custom string is Spam or Ham.
    """
    # 1. Clean the string using the exact same pre-processing
    cleaned_message = clean_text(message)
    
    # 2. Vectorize using the fitted vectorizer
    vectorized_message = vectorizer.transform([cleaned_message])
    
    # 3. Predict and classify
    prediction = model.predict(vectorized_message)[0]
    
    label = "Spam" if prediction == 1 else "Ham"
    
    print("\n--- Custom SMS Prediction ---")
    print(f"Original SMS  : '{message}'")
    print(f"Cleaned SMS   : '{cleaned_message}'")
    print(f"Classification: >>> {label} <<<")
    
    return label

def main():
    # Replace 'spam_dataset.csv' with your actual dataset file path
    dataset_path = 'spam_dataset.xlsx' 
    
    # 1. Data Pre-processing
    df = load_and_preprocess_data(dataset_path)
    
    if df.empty:
        print("Error: Empty dataset.")
        return
        
    # 2 & 3. Feature Extraction and Model Training
    model, vectorizer = train_and_evaluate_model(df)
    
    # 4. Data Visualization
    visualize_data(df)
    
    # 5. Custom Prediction Pipeline
    sample_spam = "CONGRATULATIONS! You've been selected for a $500 gift card. Reply WIN to claim."
    sample_ham = "Hey, are we still meeting up for coffee tomorrow at 10?"
    
    predict_custom_sms(sample_spam, model, vectorizer)
    predict_custom_sms(sample_ham, model, vectorizer)

if __name__ == "__main__":
    main()
