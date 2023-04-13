from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

df = pd.read_csv('bank.csv')

tfidf.fit_transform(df['text']).toarray()

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():

    text = request.form.get('text')

    # 2. vectorize
    vector_input = tfidf.transform([text]).toarray()
    # 3. predict
    result = model.predict(vector_input)[0]

    return jsonify({'target': str(result)})


if __name__ == '__main__':
    app.run()
