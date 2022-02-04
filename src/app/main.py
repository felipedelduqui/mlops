from binhex import HexBin
from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from googletrans import Translator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
import os
import pickle #to serialize/serialization (save trained model)

warnings.filterwarnings("default")

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

uri = 'https://raw.githubusercontent.com/alura-cursos/1576-mlops-machine-learning/aula-5/casas.csv'
#house_price = pd.read_csv(uri)
#df = pd.read_csv('../../house_price.csv')
#col_pt_to_en = {
#    'tamanho' : 'size',
#    'ano' : 'year',
#    'garagem' : 'park slots',
#    'preco' : 'price'
#}

model = pickle.load(open('../../models/model.sav', 'rb'))

#df = df.rename(columns = col_pt_to_en)
columns = ['size', 'year', 'parking slots']
#X = df.drop('price', axis=1)
#y = df['price']
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

#model = LinearRegression()
#model.fit(X_train, y_train)

@app.route('/')
def home():
    return "Testing API."

@app.route('/sentiment/<sentence>')
@basic_auth.required
def sentiment(sentence):
    translator = Translator()
    sentence_en = translator.translate(sentence, dest='en')
    tb_en = TextBlob(sentence_en.text)
    polarity = tb_en.polarity
    return "Sentence: {} - Polarity: {}".format(sentence, polarity)

@app.route('/predict/', methods=['POST']) #'/predict/<size_house>', if the train would involve one variable only
@basic_auth.required                      #POST will send a post type http
def predict():                              
    data = request.get_json()
    data_input = [data[col] for col in columns]
    warnings.filterwarnings("ignore")
    price = model.predict([data_input])
    #return "Predicted price: R${:.2f}".format(price[0])
    return jsonify(price=price[0])
    
#host ='0.0.0.0', since we will deploy through Docker, App Engine and local
app.run(debug=True, host='0.0.0.0') #Flask will update automatically after saving