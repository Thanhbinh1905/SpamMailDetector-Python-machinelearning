import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report

# Đọc dữ liệu từ file CSV
def load_data(file_path):
    """Đọc dữ liệu từ file CSV."""
    data = pd.read_csv(file_path)
    return data

data = load_data("dataset/email.csv")

# Phân chia dữ liệu thành features (X) và labels (y)
X = data['Message']
y = data['Category']

# Chuyển đổi văn bản thành dạng ma trận số
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Chia dữ liệu thành tập train và test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hàm để đánh giá mô hình
def evaluate_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()  # Bắt đầu tính thời gian
    model.fit(X_train, y_train)
    train_time = time.time() - start_time  # Thời gian huấn luyện

    start_time = time.time()  # Bắt đầu tính thời gian dự đoán
    y_pred = model.predict(X_test)
    test_time = time.time() - start_time  # Thời gian dự đoán

    accuracy = accuracy_score(y_test, y_pred)

    # Báo cáo chi tiết các chỉ số
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Trích xuất Precision, Recall, F1-score từ báo cáo
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    
    return accuracy, precision, recall, f1_score, train_time, test_time

# Tạo các mô hình
models = {
    'MultinomialNB': MultinomialNB(),
    'GaussianNB': GaussianNB(),
    'BernoulliNB': BernoulliNB(),
    'ComplementNB': ComplementNB()
}

# Đánh giá các mô hình
results = []
for model_name, model in models.items():
    if model_name == 'GaussianNB':  # GaussianNB yêu cầu chuyển đổi từ ma trận thưa sang mảng đặc
        accuracy, precision, recall, f1_score, train_time, test_time = evaluate_model(model, X_train.toarray(), y_train, X_test.toarray(), y_test)
    else:
        accuracy, precision, recall, f1_score, train_time, test_time = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1_score,
        'Training Time (s)': train_time,
        'Testing Time (s)': test_time
    })

# Chuyển đổi kết quả thành DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('naive_bayes_model_comparison.csv', index=False)

# # In bảng kết quả
# print("So sanh cac mo hinh Naive Bayes:")
# print(results_df)

# # Biểu đồ so sánh độ chính xác
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Model', y='Accuracy', data=results_df)
# plt.title('So sánh độ chính xác của các mô hình Naïve Bayes')
# plt.ylabel('Accuracy')
# plt.xlabel('Model')
# plt.show()

# # Biểu đồ so sánh Precision
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Model', y='Precision', data=results_df)
# plt.title('So sánh Precision của các mô hình Naïve Bayes')
# plt.ylabel('Precision')
# plt.xlabel('Model')
# plt.show()

# # Biểu đồ so sánh Recall
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Model', y='Recall', data=results_df)
# plt.title('So sánh Recall của các mô hình Naïve Bayes')
# plt.ylabel('Recall')
# plt.xlabel('Model')
# plt.show()

# # Biểu đồ so sánh F1-score
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Model', y='F1-score', data=results_df)
# plt.title('So sánh F1-score của các mô hình Naïve Bayes')
# plt.ylabel('F1-score')
# plt.xlabel('Model')
# plt.show()

# # Biểu đồ so sánh thời gian huấn luyện
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Model', y='Training Time (s)', data=results_df)
# plt.title('So sánh thời gian huấn luyện của các mô hình Naïve Bayes')
# plt.ylabel('Training Time (s)')
# plt.xlabel('Model')
# plt.show()

# # Biểu đồ so sánh thời gian dự đoán
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Model', y='Testing Time (s)', data=results_df)
# plt.title('So sánh thời gian dự đoán của các mô hình Naïve Bayes')
# plt.ylabel('Testing Time (s)')
# plt.xlabel('Model')
# plt.show()
