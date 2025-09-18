from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 初始化 Flask 应用
app = Flask(__name__)

# 准备一个简单的训练数据
training_sentences = [
    "I love this product",
    "This is amazing",
    "I feel great",
    "I hate this",
    "This is terrible",
    "I feel bad"
]
training_labels = ["positive", "positive", "positive", "negative", "negative", "negative"]

# 向量化 + 训练一个简单模型
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_sentences)
model = MultinomialNB()
model.fit(X_train, training_labels)

@app.route("/")
def home():
    return "Sentiment Analysis API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    X_test = vectorizer.transform([text])
    prediction = model.predict(X_test)[0]
    return jsonify({"text": text, "sentiment": prediction})

if __name__ == "__main__":
    app.run(debug=True)
    from flask import Flask, request, jsonify
from flask_cors import CORS   # <-- add this

app = Flask(__name__)
CORS(app)  # <-- enable CORS for all routes

@app.route('/')
def home():
    return "Sentiment Analysis API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    sentiment = "positive" if "love" in text.lower() or "happy" in text.lower() else "negative"
    return jsonify({"text": text, "sentiment": sentiment})
