import os
import re
from http import HTTPStatus
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import pickle


decisiontreeclassifier_using_cbow_vector = load(
    "decisiontreeclassifier_using_cbow_vector.joblib")
decisiontreeclassifier_using_count_vector = load(
    "decisiontreeclassifier_using_count_vector.joblib")
decisiontreeclassifier_using_skip_gram_vector = load(
    "decisiontreeclassifier_using_skip_gram_vector.joblib")
decisiontreeclassifier_using_tfidf_vector = load(
    "decisiontreeclassifier_using_tfidf_vector.joblib")
kneighborsclassifier_using_cbow_vector = load(
    "kneighborsclassifier_using_cbow_vector.joblib")
kneighborsclassifier_using_count_vector = load(
    "kneighborsclassifier_using_count_vector.joblib")
kneighborsclassifier_using_skip_gram_vector = load(
    "kneighborsclassifier_using_skip_gram_vector.joblib")
kneighborsclassifier_using_tfidf_vector = load(
    "kneighborsclassifier_using_tfidf_vector.joblib")
logisticregression_using_cbow_vector = load(
    "logisticregression_using_cbow_vector.joblib")
logisticregression_using_count_vector = load(
    "logisticregression_using_count_vector.joblib")
logisticregression_using_skip_gram_vector = load(
    "logisticregression_using_skip_gram_vector.joblib")
logisticregression_using_tfidf_vector = load(
    "logisticregression_using_tfidf_vector.joblib")
multinomialnb_using_count_vector = load(
    "multinomialnb_using_count_vector.joblib")
multinomialnb_using_tfidf_vector = load(
    "multinomialnb_using_tfidf_vector.joblib")
randomforestclassifier_using_cbow_vector = load(
    "randomforestclassifier_using_cbow_vector.joblib")
randomforestclassifier_using_count_vector = load(
    "randomforestclassifier_using_count_vector.joblib")
randomforestclassifier_using_skip_gram_vector = load(
    "randomforestclassifier_using_skip_gram_vector.joblib")
randomforestclassifier_using_tfidf_vector = load(
    "randomforestclassifier_using_tfidf_vector.joblib")
svc_using_cbow_vector = load("svc_using_cbow_vector.joblib")
svc_using_count_vector = load("svc_using_count_vector.joblib")
svc_using_skip_gram_vector = load("svc_using_skip_gram_vector.joblib")
svc_using_tfidf_vector = load("svc_using_tfidf_vector.joblib")

xgbclassifier_using_cbow_vector = load(
    "xgbclassifier_using_cbow_vector.joblib")
xgbclassifier_using_count_vector = load(
    "xgbclassifier_using_count_vector.joblib")
xgbclassifier_using_skip_gram_vector = load(
    "xgbclassifier_using_skip_gram_vector.joblib")
xgbclassifier_using_tfidf_vector = load(
    "xgbclassifier_using_tfidf_vector.joblib")


classifiers_list = [
    ("decisiontreeclassifier_using_cbow_vector",
     decisiontreeclassifier_using_cbow_vector),
    ("decisiontreeclassifier_using_count_vector",
     decisiontreeclassifier_using_count_vector),
    ("decisiontreeclassifier_using_skip_gram_vector",
     decisiontreeclassifier_using_skip_gram_vector),
    ("decisiontreeclassifier_using_tfidf_vector",
     decisiontreeclassifier_using_tfidf_vector),
    ("kneighborsclassifier_using_cbow_vector",
     kneighborsclassifier_using_cbow_vector),
    ("kneighborsclassifier_using_count_vector",
     kneighborsclassifier_using_count_vector),
    ("kneighborsclassifier_using_skip_gram_vector",
     kneighborsclassifier_using_skip_gram_vector),
    ("kneighborsclassifier_using_tfidf_vector",
     kneighborsclassifier_using_tfidf_vector),
    ("logisticregression_using_cbow_vector", logisticregression_using_cbow_vector),
    ("logisticregression_using_count_vector",
     logisticregression_using_count_vector),
    ("logisticregression_using_skip_gram_vector",
     logisticregression_using_skip_gram_vector),
    ("logisticregression_using_tfidf_vector",
     logisticregression_using_tfidf_vector),
    ("multinomialnb_using_count_vector", multinomialnb_using_count_vector),
    ("multinomialnb_using_tfidf_vector", multinomialnb_using_tfidf_vector),
    ("randomforestclassifier_using_cbow_vector",
     randomforestclassifier_using_cbow_vector),
    ("randomforestclassifier_using_count_vector",
     randomforestclassifier_using_count_vector),
    ("randomforestclassifier_using_skip_gram_vector",
     randomforestclassifier_using_skip_gram_vector),
    ("randomforestclassifier_using_tfidf_vector",
     randomforestclassifier_using_tfidf_vector),
    ("svc_using_cbow_vector", svc_using_cbow_vector),
    ("svc_using_count_vector", svc_using_count_vector),
    ("svc_using_skip_gram_vector", svc_using_skip_gram_vector),
    ("svc_using_tfidf_vector", svc_using_tfidf_vector), (
        "xgbclassifier_using_cbow_vector", xgbclassifier_using_cbow_vector),
    ("xgbclassifier_using_count_vector", xgbclassifier_using_count_vector),
    ("xgbclassifier_using_skip_gram_vector", xgbclassifier_using_skip_gram_vector),
    ("xgbclassifier_using_tfidf_vector", xgbclassifier_using_tfidf_vector), ]


with open("count_vectorizer.pickle", 'rb') as handle:
    count_vectorizer = pickle.load(handle)
with open("tfidf_vectorizer.pickle", 'rb') as handle:
    tfidf_vectorizer = pickle.load(handle)
with open("skip_gram_model.pickle", 'rb') as handle:
    skip_gram_model = pickle.load(handle)
with open("cbow_model.pickle", 'rb') as handle:
    cbow_model = pickle.load(handle)

app = Flask(__name__)
CORS(app)


def get_word_embeddings(token_list, vector, k=150):
    if len(token_list) < 1:
        return np.zeros(k)
    else:
        vectorized = [vector[word] if word in vector else np.random.rand(
            k) for word in token_list]

    sum = np.sum(vectorized, axis=0)
    return sum/len(vectorized)


@ app.route("/")
def index():
    return jsonify({'message': 'Home page', "projet": "disaster-tweets-backend-simple-models"}), 200


@ app.route("/prediction", methods=['GET', 'POST'])
def predict():
    args = request.args
    if (args and args['tweet']):
        tweet = args['tweet']
        models = args['models']
        print('\n\n')
        print("models:", len(models.split(",")))
        print('\n\n')
        try:
            predictions = []
            tweet_tfidf = tfidf_vectorizer.transform([tweet, ])
            tweet_count = count_vectorizer.transform([tweet, ])
            tweet_skip_gram = np.array([get_word_embeddings(
                tweet.split(" "), skip_gram_model)])
            tweet_cbow = np.array(
                [get_word_embeddings(tweet.split(" "), cbow_model)])

            for classifier_tuple in classifiers_list:
                classifier_full_name = classifier_tuple[0]
                classifier = classifier_tuple[1]
                if classifier_full_name in models:
                    if "count_vector" in classifier_full_name:
                        tweet = tweet_count
                    if "tfidf_vector" in classifier_full_name:
                        tweet = tweet_tfidf
                    if "skip_gram" in classifier_full_name:
                        tweet = tweet_skip_gram
                    if "cbow" in classifier_full_name:
                        tweet = tweet_cbow
                    result = classifier.predict(tweet)
                    print('result:', result)
                    isDisasterTweet = bool(result.tolist()[0])

                    predictions.append({classifier_full_name: isDisasterTweet})

            responce = {'prediction': predictions}
        except Exception as e:
            print(e)
            responce = {'message': str(e)}
    else:
        responce = {'message': "no tweet"}

    return jsonify(responce), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
