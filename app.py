from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Thay 'your_secret_key' bằng một chuỗi bí mật

# Đọc dữ liệu và huấn luyện mô hình
def load_model():
    global model, vectorizer
    try:
        with open('model.pkl', 'rb') as model_file:
            model, vectorizer = pickle.load(model_file)
    except FileNotFoundError:
        data = pd.read_csv('dataset/email.csv')
        X = data['Message']
        y = data['Category']
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(X)
        model = MultinomialNB()
        model.fit(X, y)
        with open('model.pkl', 'wb') as model_file:
            pickle.dump((model, vectorizer), model_file)

load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    spam_prob = None
    message = ""

    if request.method == "POST":
        message = request.form['message']
        message_vector = vectorizer.transform([message])
        spam_prob = model.predict_proba(message_vector)[0][1] * 100
        spam_prob = round(spam_prob, 2)
        prediction = "Spam" if spam_prob > 50 else "Ham"
        
    return render_template("index.html", prediction=prediction, spam_prob=spam_prob, message=message)

@app.route("/feedback", methods=["POST"])
def feedback():
    message = request.form['message']
    correct_classification = request.form['correct_classification']
    
    if correct_classification == "no":
        correct_label = request.form.get('correct_label', None)
        
        if correct_label:
            feedback_data = pd.DataFrame({'Category': [correct_label], 'Message': [message]})
            feedback_data.to_csv('dataset/feedback.csv', mode='a', header=False, index=False)
            update_model()
            
            flash('Thank you for your feedback!', 'success')
            
    return redirect(url_for('index'))

def update_model():
    # Đọc dữ liệu từ các tệp CSV
    training_data = pd.read_csv('dataset/email.csv').fillna('')
    feedback_data = pd.read_csv('dataset/feedback.csv').fillna('')

    # Kết hợp dữ liệu
    data = pd.concat([training_data, feedback_data], ignore_index=True)

    # Xử lý giá trị NaN
    data['Message'] = data['Message'].fillna('')  # Thay thế NaN bằng chuỗi rỗng
    data['Category'] = data['Category'].fillna('unknown')  # Thay thế NaN trong Category bằng 'unknown' (hoặc giá trị khác)

    # Huấn luyện mô hình mới
    X = data['Message']
    y = data['Category']
    X = vectorizer.fit_transform(X)
    model.fit(X, y)

    # Lưu mô hình
    with open('model.pkl', 'wb') as model_file:
        pickle.dump((model, vectorizer), model_file)


if __name__ == "__main__":
    app.run(debug=True)