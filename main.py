from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load and prepare dataset
def load_and_train_model(dataset_path):
    # Load dataset
    data = pd.read_csv(dataset_path)
    X = data['Resume']
    y = data['Category']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    visualize_dataset(data)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Vectorize the text data
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, stop_words='english')
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, penalty='l2', C=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(C=0.5, kernel='linear'),
        # Uncomment XGBoost if needed
        # "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=5)
    }

    metrics = {}
    all_labels = list(set(y_train))
    best_model_name = None
    best_accuracy = 0

    for model_name, model in models.items():
        model.fit(X_train_vect, y_train)
        predictions = model.predict(X_test_vect)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, predictions, labels=all_labels)
        cv_scores = cross_val_score(model, X_train_vect, y_train, cv=5)
        cv_mean = cv_scores.mean()

        print(f"{model_name} - Accuracy: {accuracy}, Precision: {precision}, CV Mean Accuracy: {cv_mean}")

        metrics[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "confusion_matrix": cm,
            "cv_mean_accuracy": cv_mean
        }
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_model = model

        # Plot confusion matrix for each model
        plot_confusion_matrix(cm, model_name)
    
    plot_model_comparison(metrics)    

    print(f"Best Model: {best_model_name} with CV Accuracy: {best_accuracy}")

    # Save the best model and vectorizer for later use
    joblib.dump(best_model, 'resume_classifier.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    plot_learning_curve(best_model, X_train_vect, y_train)

def plot_learning_curve(model, X_train_vect, y_train):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(model, X_train_vect, y_train, cv=5)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score', marker='o')
    plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation Score', marker='o')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid()
    plt.show()

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Call visualization for the dataset
 
def plot_model_comparison(metrics):
    model_names = list(metrics.keys())
    accuracies = [metrics[name]['accuracy'] for name in model_names]
    precisions = [metrics[name]['precision'] for name in model_names]
    cv_means = [metrics[name]['cv_mean_accuracy'] for name in model_names]

    x = range(len(model_names))

    # Create subplots for accuracy, precision, and cross-validation mean accuracy
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Plot accuracy
    axes[0].bar(x, accuracies, color='skyblue')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45)

    # Plot precision
    axes[1].bar(x, precisions, color='salmon')
    axes[1].set_title('Model Precision')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Precision')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45)

    # Plot cross-validation mean accuracy
    axes[2].bar(x, cv_means, color='lightgreen')
    axes[2].set_title('Model Cross-Validation Mean Accuracy')
    axes[2].set_xlabel('Model')
    axes[2].set_ylabel('CV Mean Accuracy')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(model_names, rotation=45)

    plt.tight_layout()
    plt.show()
 

def visualize_dataset(data):
    # Distribution of categories
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Category', data=data, order=data['Category'].value_counts().index)
    plt.xticks(rotation=90)
    plt.title('Distribution of Resume Categories')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Word cloud for resume text data
    all_text = ' '.join(data['Resume'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Resume Text Data')
    plt.tight_layout()
    plt.show()

    # Show the distribution of resume lengths
    data['resume_length'] = data['Resume'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(data['resume_length'], bins=30, kde=True)
    plt.title('Distribution of Resume Lengths')
    plt.xlabel('Number of Words in Resume')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Category vs. Average resume length
    plt.figure(figsize=(12, 6))
    avg_length_per_category = data.groupby('Category')['resume_length'].mean().sort_values()
    sns.barplot(x=avg_length_per_category.index, y=avg_length_per_category.values)
    plt.xticks(rotation=90)
    plt.title('Average Resume Length per Category')
    plt.xlabel('Category')
    plt.ylabel('Average Number of Words')
    plt.tight_layout()
    plt.show()

# Load the trained model and vectorizer
def load_model():
    model = joblib.load('resume_classifier.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, vectorizer, label_encoder

# Function to classify resume category
def classify_resume(resume_text, model, vectorizer, label_encoder):
    resume_vect = vectorizer.transform([resume_text])
    category_encoded = model.predict(resume_vect)[0]
    category = label_encoder.inverse_transform([category_encoded])[0]
    return category

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

@app.route("/")
def matchresume():
    return render_template('matchresume.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')

        resumes = []
        categories = []
        model, vectorizer, label_encoder = load_model()
        for resume_file in resume_files:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)
            resume_text = extract_text(filename)  # Make sure to define extract_text function
            resumes.append(resume_text)
            category = classify_resume(resume_text, model, vectorizer, label_encoder)
            categories.append(category)

        if not resumes or not job_description:
            return render_template('matchresume.html', message="Please upload resumes and enter a job description.")

        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()

        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        top_indices = similarities.argsort()[-2:][::-1]
        top_resumes = [(resume_files[i].filename, categories[i]) for i in top_indices]
        similarity_scores = [round(similarities[i], 2) for i in top_indices]
        matched_categories = [categories[i] for i in top_indices]

        return render_template('matchresume.html', message="Top matching resumes:", 
                               top_resumes=top_resumes, similarity_scores=similarity_scores, 
                               matched_categories=matched_categories)

    return render_template('matchresume.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    load_and_train_model('UpdatedResumeDataSet.csv')
    app.run(debug=False)
