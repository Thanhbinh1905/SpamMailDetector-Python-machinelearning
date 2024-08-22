import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

#read the CSV file
def load_data(file_path):
    """Đọc dữ liệu từ file CSV."""
    data = pd.read_csv(file_path)
    return data
datatraining = load_data("dataset/feedback.csv")
data = load_data("dataset/email.csv")

print(datatraining.info())

# Bước 2: Phân chia dữ liệu thành features (X) và labels (y)
X = data['Message']
y = data['Category']

# Bước 3: Chuyển đổi văn bản thành dạng ma trận số
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Bước 4: Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 5: Tạo và huấn luyện mô hình Naïve Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Bước 6: Dự đoán trên tập test
y_pred = model.predict(X_test)

# # Bước 7: Kiểm tra độ chính xác và in báo cáo
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
