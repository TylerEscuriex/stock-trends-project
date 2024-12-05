import google.generativeai as genai
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

# Ensure the Gemini API key is correctly set
api_key = "AIzaSyAQBhCNYN_LXsqK1miOLaNJPl5KW2Lt5vk"
genai.configure(api_key=api_key)

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
    """Fetch key financial data for the stock ticker using yfinance."""
    stock = yf.Ticker(ticker)
    stock_info = stock.info  # This provides a dictionary of financial information
    
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

    # Filter out None values
    return {key: value for key, value in stockdata.items() if value is not None} or None

def generate_gemini_recommendation(ticker, headlines_sentiment, stockdata):
    """Generate stock recommendation using Gemini."""
    try:
        if not ticker:
            raise ValueError("Ticker symbol is required.")
        if not headlines_sentiment or not isinstance(headlines_sentiment, list):
            raise ValueError("Headlines and sentiments must be a non-empty list.")
        if not stockdata or not isinstance(stockdata, dict):
            raise ValueError("Stock data must be a non-empty dictionary.")

        headlines_formatted = "\n".join([f"- {headline}: {sentiment}" for headline, sentiment in headlines_sentiment])
        stockdata_formatted = "\n".join([f"- {key}: {value}" for key, value in stockdata.items()])

        prompt = (
            f"Given the following stock data for {ticker}:\n\n"
            f"Headlines and Sentiments:\n{headlines_formatted}\n\n"
            f"Financial Data:\n{stockdata_formatted}\n\n"
            "Based on this data, should an investor consider buying, holding, or selling this stock? "
            "Provide a clear and concise recommendation."
        )

        # Call Gemini to generate text
        response = genai.generate_text(prompt=prompt)

        # Extract the recommendation text
        return response.text if response else "No recommendation generated."
    
    except Exception as e:
        # Log or flash the error for debugging
        print(f"Error generating Gemini recommendation: {e}")
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
                # Generate the recommendation
                recommendation = generate_gemini_recommendation(ticker, headlines_sentiment, stockdata)
                flash(f'Financial data retrieved for {ticker}', category='success')
            else:
                flash(f'No financial data found for {ticker}. Some data might be unavailable for this stock.', category='warning')

    return render_template(
        "home.html",
        headlines=headlines_sentiment,
        ticker=ticker,
        stockdata=stockdata or {},
        recommendation=recommendation,
        user=current_user
    )

@views.route('/about', methods=['GET'])
@login_required
def about():
    return render_template("about.html", user=current_user)
