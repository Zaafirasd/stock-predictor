import json
import base64
import io
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def handler(event, context):
    # --- 1. Get Ticker from Request ---
    try:
        body = json.loads(event.get('body', '{}'))
        # Ensure only the first ticker is processed to prevent errors
        ticker = body.get('ticker', '').strip().upper().split()[0]
        if not ticker:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Ticker symbol cannot be empty.'})
            }
    except Exception as e:
        return {'statusCode': 400, 'body': json.dumps({'error': f'Invalid request format: {str(e)}'})}

    try:
        # --- 2. Fetch Historical Data ---
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365) # 5 years of data
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if stock_data.empty:
            return {
                'statusCode': 404,
                'body': json.dumps({'error': f"No data found for ticker '{ticker}'. Please check the symbol."})
            }
        
        stock_data.reset_index(inplace=True)
        stock_data['DateOrdinal'] = stock_data['Date'].map(datetime.toordinal)

        # --- 3. Train Linear Regression Model ---
        X = stock_data[['DateOrdinal']]
        y = stock_data['Close']
        model = LinearRegression()
        model.fit(X, y)

        # --- 4. Predict Future Prices ---
        last_date = stock_data['Date'].iloc[-1]
        future_dates = pd.to_datetime([last_date + timedelta(days=x) for x in range(1, 366)])
        future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        future_predictions = model.predict(future_ordinals)

        # --- 5. Generate Recommendation ---
        last_actual_price = y.iloc[-1]
        predicted_price_in_30_days = future_predictions[29]

        if predicted_price_in_30_days > last_actual_price * 1.02:
            recommendation = "BUY"
        elif predicted_price_in_30_days < last_actual_price:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"

        # --- 6. Create Plot ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(stock_data['Date'], stock_data['Close'], label='Historical Price', color='royalblue', linewidth=2)
        ax.plot(future_dates, future_predictions, label='Prediction', color='darkorange', linestyle='--')
        
        ax.set_title(f'Stock Price Forecast for {ticker}', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.legend()
        fig.tight_layout()

        # --- 7. Convert Plot to Base64 String ---
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # --- 8. Return JSON Response ---
        return {
            'statusCode': 200,
            'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps({
                'ticker': ticker,
                'recommendation': recommendation,
                'plot_url': plot_base64
            })
        }

    except Exception as e:
        # Generic error handler for any other issues
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'An unexpected error occurred: {str(e)}'})
        }
