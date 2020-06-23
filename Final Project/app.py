from flask import Flask, render_template
from flask import request
import pickle

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        article = request.form.get['article']
    result={}
    result["message"]=" "
    if request.method == "POST":
        loaded_vectorizer=pickle.load(open('vectorizer.sav', 'rb'))
        loaded_model=pickle.load(open('Bnb_model.sav', 'rb'))
        test_sample=loaded_vectorizer.transform([article])
        pred=loaded_model.predict(test_sample.toarray())[0]
        if pred==0:
            result["message"]="This article is fake!"
        else: 
            result["message"]="This article is real!"
        print(article)
        print(result)
    return render_template('index.html',result=result)

@app.route('/response', methods=["POST"])
def response(): 
    result={}
    article=request.form.get("article")
    loaded_vectorizer=pickle.load(open('vectorizer.sav', 'rb'))
    loaded_model=pickle.load(open('Bnb_model.sav', 'rb'))
    test_sample=loaded_vectorizer.transform([article])
    print(test_sample)
    print(loaded_model.predict(test_sample.toarray()))
    pred=loaded_model.predict(test_sample.toarray())[0]
    print(pred)
    if pred=='1':
        result["message"]="This article is real!"
    else: 
        result["message"]="This article is fake!"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run()

