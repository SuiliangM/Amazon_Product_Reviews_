import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

review_data_file = './Reviews.csv'
reviews_df = pd.read_csv(filepath_or_buffer = review_data_file, header = 0)

def data_preprocessing(reviews_df : pd.DataFrame) -> pd.DataFrame:

    stop_words = set(stopwords.words('english'))

    for index, row in reviews_df.fillna('').iterrows():
        sent = row["Summary"]
        # print(f"printing sentence {sent} at id {row['Id']}")
        # retrieve the word_tokens
        word_tokens = word_tokenize(sent)
        
        # remove stop words
        sentence_with_stop_words_removed = [w for w in word_tokens if not w.lower() in stop_words]

        # stemming
        stemming_result = []
        stemming_result_string = ""
        ps = PorterStemmer()
        for w in sentence_with_stop_words_removed:
            stemming_result.append(ps.stem(w))
        
        stemming_result_string = " ".join(stemming_result)
        
        reviews_df.at[index, 'processed_summary'] = stemming_result_string

    reviews_df["predicted_result"] = None
    return reviews_df

def create_unigram_model(reviews_df: pd.DataFrame):
    # Create the CountVectorizer instance with unigram model
    # limit the max_features to be 1000 in order not to get memory error
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=1000)
    
    documents = []
    for _, row in reviews_df.iterrows():
        documents.append(row["processed_summary"])

    # Fit and transform the text data
    # X is a sprase matrix
    X = vectorizer.fit_transform(documents)

    # Convert the result to an array
    # comment out the following line because it leads to memory error
    # unigram_matrix = X.toarray()

    # Get feature names to see which words correspond to which columns
    # feature_names = vectorizer.get_feature_names_out()
    
    # print("Feature Names:\n", feature_names)
    
    return X

def create_bigram_model(reviews_df: pd.DataFrame):
    # Create the CountVectorizer instance with bigram model
    # limit the max_features to be 1000 in order not to get memory error
    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=1000)
    
    documents = []
    for _, row in reviews_df.iterrows():
        documents.append(row["processed_summary"])

    # Fit and transform the text data
    # X is a sprase matrix
    X = vectorizer.fit_transform(documents)

    # Convert the result to an array
    # comment out the following line because it leads to memory error
    # unigram_matrix = X.toarray()

    # Get feature names to see which words correspond to which columns
    # feature_names = vectorizer.get_feature_names_out()
    
    # print("Feature Names:\n", feature_names)
    
    return X
    

def model_baseline_logistics(X, reviews_df):
    # obtain the labels so we can do train - test split for the given dataset
    labels = []
    for _, row in reviews_df.iterrows():
        if row["Score"] > 3:
            # appending a positive for the semantics if score is larger than 3
            labels.append(1)
        else:
            labels.append(0)
    
    # apply train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = clf.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report

    
    
reviews_df = data_preprocessing(reviews_df)
X_unigram = create_unigram_model(reviews_df)
X_unigram_accuracy, X_unigram_report = model_baseline_logistics(X_unigram, reviews_df)
X_bigram = create_bigram_model(reviews_df)
X_bigram_accuracy, X_bigram_report = model_baseline_logistics(X_bigram, reviews_df)
print("Unigram Accuracy:", X_unigram_accuracy)
print("Bigram Accuracy: ", X_bigram_accuracy)
print("Unigram Classification Report:\n", X_unigram_report)
print("Bigram Classification Report:\n", X_bigram_report)
