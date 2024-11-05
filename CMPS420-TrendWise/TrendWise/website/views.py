import openai
import requests
from bs4 import BeautifulSoup
from flask import Blueprint, render_template, request, flash
from flask_login import login_required, current_user
from transformers import pipeline
import yfinance as yf
import os




views = Blueprint('views', __name__)

# Load the Hugging Face FinBERT model for sentiment analysis
nlp_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Set your OpenAI API key (ensure this is securely stored in production)
openai.api_key = 'your_openai_api_key_here'

def scrape_headlines(ticker):
    base_url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch"
    response = requests.get(base_url)
    
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    headlines = []
    
    for item in soup.find_all('a', class_='subtle-link fin-size-small thumb yf-1e4diqp'):
        headline = item.get('aria-label')
        if headline:
            headlines.append(headline)
    
    return headlines

def get_stockdata(ticker):
    """
    Fetches key financial data for a given stock ticker using yfinance.
    """
    stock = yf.Ticker(ticker)
    stock_info = stock.info  # This provides a dictionary of financial information
    
    # Select relevant financial data
    stockdata = {
        "Market Cap": stock_info.get("marketCap"),
        "Price-to-Earnings Ratio (P/E)": stock_info.get("trailingPE"),
        "Price-to-Book Ratio (P/B)": stock_info.get("priceToBook"),
        "Dividend Yield": stock_info.get("dividendYield"),
        "52-Week High": stock_info.get("fiftyTwoWeekHigh"),
        "52-Week Low": stock_info.get("fiftyTwoWeekLow"),
        "Beta": stock_info.get("beta"),
        "Revenue": stock_info.get("totalRevenue"),
        "Net Income": stock_info.get("netIncomeToCommon"),
    }

    # Filter out any None values from stockdata
    stockdata = {key: value for key, value in stockdata.items() if value is not None}
    
    return stockdata if stockdata else None


def generate_gpt_recommendation(ticker, headlines_sentiment, stockdata):
    try:
        # Prepare the data for the API
        prompt = f"Given the following stock data for {ticker}:\n\n" \
                 f"Headlines and Sentiments:\n" + \
                 "\n".join([f"{headline}: {sentiment}" for headline, sentiment in headlines_sentiment]) + \
                 "\n\nFinancial Data:\n" + \
                 "\n".join([f"{key}: {value}" for key, value in stockdata.items()]) + \
                 "\n\nBased on this data, should an investor consider buying, holding, or selling this stock? Provide a clear recommendation."

        # Corrected method call
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract the generated response
        recommendation = response['choices'][0]['message']['content']
        return recommendation

    except Exception as e:
        # Log the exact error
        print(f"Error generating GPT recommendation: {e}")
        return "Could not generate a recommendation due to an error." 
    
@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    ticker = None
    headlines_sentiment = None
    stockdata = None
    recommendation = None

    if request.method == 'POST':
        ticker = request.form.get('ticker').upper()

        if len(ticker) < 1:
            flash('Please enter a valid stock ticker!', category='error')
        else:
            headlines = scrape_headlines(ticker)
            stockdata = get_stockdata(ticker)

            if headlines:
                headlines_sentiment = [(headline, nlp_pipeline(headline)[0]['label']) for headline in headlines]
                flash(f'Found and analyzed headlines for {ticker}', category='success')
            
            if stockdata:
                recommendation = generate_gpt_recommendation(ticker, headlines_sentiment, stockdata)
                flash(f'Financial data retrieved for {ticker}', category='success')
            else:
                flash(f'No financial data found for {ticker}. Some data might be unavailable for this stock.', category='warning')

    return render_template(
        "home.html",
        headlines=headlines_sentiment,
        ticker=ticker,
        stockdata=stockdata or {},  # Pass an empty dictionary if no stockdata
        recommendation=recommendation,
        user=current_user
    )

