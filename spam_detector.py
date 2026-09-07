import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def main():
    # Force UTF-8 encoding for Windows terminals to print emojis correctly
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        
    # Replace with the path to your dataset (assuming a CSV format as requested)
    file_path = 'spam_dataset.csv' 
    
    # ---------------------------------------------------------
    # Task 1 — Load and Explore the Dataset
    # ---------------------------------------------------------
    print("=== Task 1: Load and Explore the Dataset ===")
    try:
        # Load the CSV
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Fallback to the .xlsx file you have in the directory
        try:
            df = pd.read_excel('spam_dataset.xlsx')
            print("Loaded 'spam_dataset.xlsx' instead of CSV.")
        except FileNotFoundError:
            print(f"Error: {file_path} not found. Please ensure the file exists in the directory.")
            return

    # Display the first 5 rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # Print the number of rows and columns
    print(f"\nNumber of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    # Print the count of Ham vs. Spam messages
    print("\nCount of Ham vs. Spam messages:")
    print(df['Category'].value_counts())

    # ---------------------------------------------------------
    # Task 2 — Visualize the Dataset
    # ---------------------------------------------------------
    print("\n=== Task 2: Visualize the Dataset ===")
    plt.figure(figsize=(6, 4))
    df['Category'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Number of Spam vs. Ham Messages")
    plt.xlabel("Message Type")
    plt.ylabel("Frequency")
    plt.xticks(rotation=0)
    plt.tight_layout()
    # Using savefig instead of show() to avoid blocking execution in background runs
    plt.savefig('category_distribution.png') 
    print("Bar chart saved as 'category_distribution.png'")
    
    print("\nQuestion: Which type of message is more common in the dataset?")
    if df['Category'].value_counts().idxmax().lower() == 'ham':
        print("Answer: 'ham' messages are more common in the dataset.")
    else:
        print("Answer: 'spam' messages are more common in the dataset.")

    # ---------------------------------------------------------
    # Task 3 — Prepare the Data
    # ---------------------------------------------------------
    print("\n=== Task 3: Prepare the Data ===")
    # Convert Category labels into numerical values (Ham = 0, Spam = 1)
    df['Category'] = df['Category'].map(lambda x: 1 if str(x).lower() == 'spam' else 0)
    
    # Separate the data into X (Message) and y (Category)
    X = df['Message'].astype(str)
    y = df['Category']
    print("Data successfully converted. Separated into X (Message) and y (Category).")

    # ---------------------------------------------------------
    # Task 4 — Split the Dataset
    # ---------------------------------------------------------
    print("\n=== Task 4: Split the Dataset ===")
    # We split the dataset into training and testing sets to evaluate how well our model 
    # generalizes to new, unseen data, which helps prevent overfitting on the training data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dataset divided into 80% training data and 20% testing data.")

    # ---------------------------------------------------------
    # Task 5 — Convert Text into Numbers
    # ---------------------------------------------------------
    print("\n=== Task 5: Convert Text into Numbers ===")
    # We cannot directly feed text to a Machine Learning model because algorithms 
    # rely on mathematical computations (like distances, weights, and probabilities) 
    # which require numerical input rather than raw strings.
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    print("Text messages converted into numerical features using CountVectorizer (stopwords removed).")

    # ---------------------------------------------------------
    # Task 6 — Train a Machine Learning Model
    # ---------------------------------------------------------
    print("\n=== Task 6: Train a Machine Learning Model ===")
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)
    print("LogisticRegression model successfully trained.")

    # ---------------------------------------------------------
    # Task 7 — Make Predictions
    # ---------------------------------------------------------
    print("\n=== Task 7: Make Predictions ===")
    # Predict the labels of the test data and store them in a variable named exactly 'predictions'
    predictions = model.predict(X_test_vectorized)
    print("Predictions made on the test set and stored in the 'predictions' variable.")

    # ---------------------------------------------------------
    # Task 8 — Evaluate the Model
    # ---------------------------------------------------------
    print("\n=== Task 8: Evaluate the Model ===")
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Generate and display a Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham (0)', 'Spam (1)'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix generated and saved as 'confusion_matrix.png'")
    
    # Calculate exactly how many messages were classified correctly (Sum of the diagonal in CM)
    correctly_classified = cm[0, 0] + cm[1, 1]
    print(f"\nThe model classified exactly {correctly_classified} messages correctly out of {len(y_test)} total test messages.")

    # ---------------------------------------------------------
    # Task 9 — Test Your Own SMS
    # ---------------------------------------------------------
    print("\n=== Task 9: Test Your Own SMS ===")
    while True:
        try:
            print("\nType 'quit' or press CTRL+C to exit.")
            user_sms = input("Enter a new SMS message to classify: ")
            
            if user_sms.strip().lower() == 'quit':
                print("Exiting prediction loop...")
                break
            
            if not user_sms.strip():
                continue
            
            # Transform using the fitted CountVectorizer
            transformed_input = vectorizer.transform([user_sms])
            
            # Predict the outcome
            prediction = model.predict(transformed_input)[0]
            
            # Print exact required output
            if prediction == 0:
                print("📩 HAM MESSAGE")
            else:
                print("🚨 SPAM MESSAGE")
                
        except (EOFError, KeyboardInterrupt):
            print("\nExiting script.")
            break

if __name__ == "__main__":
    main()
