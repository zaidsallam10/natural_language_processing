import pickle
from flask import Flask, jsonify, request, render_template
import flask

app = Flask(__name__)

def predict_sentiment(sentence):
    try:
        loaded_model = pickle.load(open('/home/xina/Desktop/Zaidar/Zaid-NLP/turkish-sentiment-analysis/turkce_sentiment_analysis.sav', 'rb'))
        print(loaded_model.predict(['lakin beklediğim kadar değil, sağ tuşta biraz zorlanmakla beraber alınıp kullanılabilir']))
        print(loaded_model.predict(['çok kötü']))
        print(loaded_model.predict(['guzel']))
        print(loaded_model.predict(['seni seviyorum']))
        print(loaded_model.predict(['senden nefret ediyorum']))
        print(loaded_model.predict(['çok kötü hizmet']))
        return {"sentence":str(sentence), "sentiment_prediction":str(loaded_model.predict([sentence])), "predicted_tag": "Positive" if  loaded_model.predict([sentence])[0]==1 else "Negative" }
    except:
        print("Model isnt there")


@app.route("/predict", methods=['POST'])
def predict():
    sentence = request.get_json()['sentence']
    response = {}
    response["response"]=predict_sentiment(sentence)
    return flask.jsonify(response)



if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=8000)