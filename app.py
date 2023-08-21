import pandas as pd
import joblib
import pickle
from flask import Flask, render_template, url_for, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from joblib import dump, load

app = Flask(__name__)

#ML Code
    
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods = ['POST'])
def predict():
    df = pd.read_csv("data/Youtube01-Psy.csv")
    df_data = df[['CONTENT', 'CLASS']]
    #Features and labels:
    df_x = df_data['CONTENT']
    df_y = df_data['CLASS']

    #Extract the feature with countVectorizer
    corpus = df_x
    cv = CountVectorizer()
    cv_x = cv.fit_transform(corpus)
    X_Train, X_Test, Y_train, Y_Test = train_test_split(cv_x, df_y, test_size= 0.33, random_state= 42)

    #Naive Bayes
    clf = MultinomialNB()
    clf.fit(X_Train, Y_train) 
    clf.score(X_Test, Y_Test)

    #Save Model
    joblib.dump(clf, 'model.pkl')
    print('Model dumped')
    clf = joblib.load('model.pkl')

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_pred = clf.predict(vect)

    return render_template('result.html', prediction = my_pred)


if __name__ == '__main__':
    app.run(debug=True)
     