# Text Classification & Similarity Analysis

## Overview

This project explores **text classification** and **similarity analysis** using various machine learning techniques. The primary focus is on analyzing the **Microsoft Research Paraphrase Corpus (MSRPC)** to identify and classify paraphrases.

## Project Structure

The repository consists of the following files:

```
📂 Exploring-Text-Classification-&-Similarity-Analysis
├── main.ipynb  # Jupyter Notebook
├── msr_paraphrase_train.txt  # Training dataset
├── msr_paraphrase_test.txt  # Test dataset
├── link.txt  # Reference links (if applicable)
```

## Requirements

Ensure you have the following dependencies installed:

- **Python 3.7+**
- **Jupyter Notebook**
- **pandas**
- **scikit-learn**
- **nltk**
- **gensim**
- **networkx**

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/text-classification-similarity-analysis.git
   cd text-classification-similarity-analysis
   ```

2. Install required dependencies:

   ```sh
   pip install pandas scikit-learn nltk gensim networkx
   ```

3. Download necessary NLTK resources:

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## Usage

1. Open the Jupyter Notebook:
   ```sh
   jupyter notebook Exploring-Text-Classification-&-Similarity-Analysis.ipynb
   ```
2. Follow the steps in the notebook to preprocess the data, train the model, and evaluate results.

## Key Methodologies

### Data Preprocessing

Preprocessing steps applied to the text data include:

- **Tokenization**
- **Lowercasing**
- **Removing non-alphanumeric characters**
- **Stopword removal**

Example preprocessing function:

```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stopwords.words('english')]
    return ' '.join(tokens)
```

### Feature Extraction

**TF-IDF (Term Frequency-Inverse Document Frequency)** is used for feature extraction:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
train_tfidf = tfidf_vectorizer.fit_transform(df_train['Preprocessed'])
test_tfidf = tfidf_vectorizer.transform(df_test['Preprocessed'])
```

### Similarity Analysis

**Cosine similarity** is used to measure the similarity between text pairs:

```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(test_tfidf, train_tfidf)
```

### Graph Construction

A graph is built to identify closely related text pairs using **NetworkX**:

```python
import networkx as nx

def construct_graph(similarity_matrix, threshold=0.8):
    graph = nx.Graph()
    num_test, num_train = similarity_matrix.shape
    for i in range(num_test):
        for j in range(num_train):
            if similarity_matrix[i][j] > threshold:
                graph.add_edge(f"test_{i}", f"train_{j}", weight=similarity_matrix[i][j])
    return graph
```

### Model Training and Evaluation

A **Logistic Regression** model is used for classification:

```python
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()
logistic.fit(train_tfidf, y_train)

train_pred = logistic.predict(train_tfidf)
test_pred = logistic.predict(test_tfidf)
```

#### Performance Evaluation

Evaluation metrics include accuracy and classification report:

```python
from sklearn.metrics import accuracy_score, classification_report

print("Training Set:")
print("Accuracy:", accuracy_score(y_train, train_pred))
print("Classification Report:")
print(classification_report(y_train, train_pred))

print("\nTest Set:")
print("Accuracy:", accuracy_score(y_test, test_pred))
print("Classification Report:")
print(classification_report(y_test, test_pred))
```

## Results

- The classification accuracy and similarity scores are presented in the notebook.
- Graph-based analysis helps visualize paraphrase relationships.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Microsoft Research** for providing the **Paraphrase Corpus**.
- Open-source contributors of **NLTK, Scikit-learn, Gensim, and NetworkX**.

For questions or contributions, please open an issue or submit a pull request on the project's **GitHub repository**.

