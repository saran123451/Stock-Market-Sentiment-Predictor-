from flask import Flask, render_template, request
import pickle
from sklearn.pipeline import Pipeline


app = Flask(__name__)

# Load the trained model (assuming you've saved it as model.pkl)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_text(text):
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    nltk.download('stopwords')
    nltk.download('punkt')
    
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation and numbers
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        headline = request.form['headline']
        processed_headline = preprocess_text(headline)
        prediction = model.predict([processed_headline])[0]
        sentiment = "Good" if prediction == 1 else "Bad"
        return render_template('index.html', headline=headline, sentiment=sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
