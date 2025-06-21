import re 
import nltk
import spacy
nltk.download('stopwords')
from nltk.corpus import stopwords
nlp = spacy.load('en_core_web_md')

def data_preprocessing(text):
  text = text.lower()
  text = re.sub(r'[^a-z\s]', '',text)
  doc = nlp(text)
  tokens = [token.text for token in doc if token.text not in stopwords.words('english')]
  return " ".join(tokens)
