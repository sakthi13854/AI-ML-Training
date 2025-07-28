from flask import Flask,request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('C:/Users/Sakthi/Downloads/model.pkl', 'rb'))

@app.route('/')
def home():  # put application's code here
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    val1 = float(request.form['feature1'])
    val2 = float(request.form['feature2'])
    val3 = float(request.form['feature3'])
    val4 = float(request.form['feature4'])


    input_data = [[val1, val2], [val3, val4]]
    predictions = model.predict(input_data)
    return render_template('result.html', prediction=predictions)
if __name__ == '__main__':
    app.run(debug=True)





