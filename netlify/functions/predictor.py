import json
import base64
import io
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Matplotlib setup for server-side rendering (no GUI)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def handler(event, context):
    """
    This is the main function Netlify will run.
    It processes the request, gets the prediction, and returns the result.
    """
    try:
        # Parse the input from the request body
        body = json.loads(event.get('body', '{}'))
        ticker_input = body.get('ticker', '').strip()

        if not ticker_input:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Please provide a stock ticker.'})
            }
        
        # Take only the first ticker if multiple are entered
        ticker = ticker_input.split()[0].upper()

        # --- Core Prediction Logic ---
        start_date = "2020-01-01"
        end_date = datetime.now().strftime('%Y-%m-%d')
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if stock_data.empty:
            raise ValueError(f"Could not find data for ticker '{ticker}'.")

        stock_data.reset_index(inplace=True)
        stock_data['DateOrdinal'] = stock_data['Date'].map(datetime.toordinal)

        X = stock_data[['DateOrdinal']]
        y = stock_data['Close']
        model = LinearRegression()
        model.fit(X, y)

        last_date = stock_data['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=365*2)
        future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        future_predictions = model.predict(future_ordinals).flatten()
        
        future_df = pd.DataFrame({'Date': future_dates, 'PredictedClose': future_predictions})

        last_actual_price = y.iloc[-1]
        predicted_price_in_30_days = future_predictions[29]
        rec_text = "HOLD"
        if predicted_price_in_30_days > last_actual_price * 1.03:
            rec_text = "BUY"
        elif predicted_price_in_30_days < last_actual_price:
            rec_text = "SELL"

        # --- Generate and Encode Plot ---
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(stock_data['Date'], stock_data['Close'], label='Historical Price', color='#007bff', linewidth=2)
        ax.plot(future_df['Date'], future_df['PredictedClose'], label='Linear Prediction', linestyle='--', color='#ffab40')
        ax.set_title(f'Stock Price Forecast for {ticker}', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Stock Price (USD)', fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close(fig)
        img_buffer.seek(0)
        plot_url = base64.b64encode(img_buffer.getvalue()).decode('utf8')

        # Prepare the successful response
        response_data = {
            'ticker': ticker,
            'recommendation': rec_text,
            'plot_url': plot_url
        }
        
        return {
            'statusCode': 200,
            'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps(response_data)
        }

    except Exception as e:
        # Return an error response if anything goes wrong
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

