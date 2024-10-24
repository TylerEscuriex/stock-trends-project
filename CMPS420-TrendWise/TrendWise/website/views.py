from flask import Blueprint, render_template, request, flash
from flask_login import login_required, current_user
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

views = Blueprint('views', __name__)

# Load the Hugging Face FinBERT model
nlp_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def scrape_headlines(ticker):
    """
    Scrapes recent headlines for a given stock ticker from Yahoo Finance.
    """
    base_url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch"
    response = requests.get(base_url)
    
    if response.status_code != 200:
        return None  # Return None if there's an issue with the request
    
    soup = BeautifulSoup(response.content, 'html.parser')
    headlines = []
    
    # Extract headlines from Yahoo Finance
    for item in soup.find_all('a', class_='subtle-link fin-size-small thumb yf-1e4diqp'):
        headline = item.get('aria-label')
        if headline:  # Check if the headline exists
            headlines.append(headline)
    
    return headlines

def scrape_stockdata(ticker):
    """
    Scrapes 
    """
    base_url = f"https://finance.yahoo.com/quote/{ticker}&/key-statistics/"
    response = requests.get(base_url)
    
    if response.status_code != 200:
        return None  # Return None if there's an issue with the request
    
    soup = BeautifulSoup(response.content, 'html.parser')
    stockdata = []
    
    # Extract headlines from Yahoo Finance
    for item in soup.find_all('table', class_='table yf-1erjfbb'):
        headline = item.get('aria-label')
        if headline:  # Check if the headline exists
            stockdata.append(headline)
    
    return stockdata

@views.route('/', methods=['GET', 'POST'])
@login_required  # Ensure the user is logged in
def home():
    ticker = None
    headlines_sentiment = None

    if request.method == 'POST':
        ticker = request.form.get('ticker').upper()  # Get the stock ticker input

        if len(ticker) < 1:
            flash('Please enter a valid stock ticker!', category='error')
        else:
            headlines = scrape_headlines(ticker)
            stockdata = scrape_stockdata(ticker)

            if headlines:
                # Perform sentiment analysis on each headline
                headlines_sentiment = [(headline, nlp_pipeline(headline)[0]['label']) for headline in headlines]
                flash(f'Found and analyzed headlines for {ticker}', category='success')
                print(stockdata)
            else:
                flash(f'No headlines found for {ticker}. Please try another stock.', category='error')

    return render_template("home.html", headlines=headlines_sentiment, ticker=ticker, user=current_user)
