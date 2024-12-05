import matplotlib
import openai
import requests
from bs4 import BeautifulSoup
from flask import Blueprint, render_template, request, flash
from flask_login import login_required, current_user
from transformers import pipeline
import yfinance as yf
import os
import pandas as pd
import google.generativeai as genai
import io
import base64
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
from dotenv import load_dotenv


views = Blueprint('views', __name__)
load_dotenv()

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
    stock = yf.Ticker(ticker)
    stock_info = stock.info  # Fetch financial information
    
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

    return {key: value for key, value in stockdata.items() if value is not None}


def generategpt_recommendation(ticker, headlines_sentiment, stockdata):
    """Generate a stock recommendation using the Gemini API."""
    try:
        prompt = f"Given the following stock data for {ticker}:\n\n" \
                 f"Headlines and Sentiments:\n" + \
                 "\n".join([f"{headline}: {sentiment}" for headline, sentiment,  in headlines_sentiment]) + \
                 "\n\nFinancial Data:\n" + \
                 "\n".join([f"{key}: {value}" for key, value in stockdata.items()]) + \
                 "\n\nBased on this data, should an investor consider buying, holding, or selling this stock? Provide a concise reccomendation, then give your reasoning."

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"Error generating recommendation: {e}")
        return "Could not generate a recommendation due to an error."


def get_stock_history(ticker):
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1mo")
        return history.reset_index().to_dict(orient='records')
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return None


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    ticker = None
    headlines_sentiment = None
    stockdata = None
    recommendation = None
    history_data = None
    graph_image = None

    if request.method == 'POST':
        ticker = request.form.get('ticker').upper()

        if not ticker:
            flash('Please enter a valid stock ticker!', category='error')
        else:
            headlines = scrape_headlines(ticker)
            stockdata = get_stockdata(ticker)
            history_data = get_stock_history(ticker)

            # Generate graph for stock price history
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="3mo")

                plt.figure(figsize=(12, 6))
                plt.plot(data.index, data['Close'], label=f"{ticker} Close Prices")
                plt.title(f"{ticker} Stock Prices - Last Month", color="#A9D18E")
                plt.xlabel("Date", color="#A9D18E")
                plt.ylabel("Price (USD)", color="#A9D18E")
                plt.grid(color="gray", linestyle="--", linewidth=0.5)
                plt.legend()
                plt.gca().set_facecolor("#121212")
                plt.gcf().set_facecolor("#1B1B1B")
                plt.tick_params(colors="#A9D18E")

                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                graph_image = base64.b64encode(img.getvalue()).decode('utf-8')
                img.close()
                plt.close()
            except Exception as e:
                print(f"Error generating graph: {e}")
                flash("Could not generate stock graph. Please try again.", category='warning')

            if headlines:
                headlines_sentiment = [(headline, nlp_pipeline(headline)[0]['label']) for headline in headlines]
                flash(f'Found and analyzed headlines for {ticker}', category='success')

            if stockdata:
                recommendation = generategpt_recommendation(ticker, headlines_sentiment, stockdata)
                flash(f'Financial data retrieved for {ticker}', category='success')
            else:
                flash(f'No financial data found for {ticker}. Some data might be unavailable for this stock.', category='warning')

    return render_template(
        "home.html",
        headlines=headlines_sentiment,
        ticker=ticker,
        stockdata=stockdata or {},
        recommendation=recommendation,
        history_data=history_data,
        graph_image=graph_image,
        user=current_user
    )
