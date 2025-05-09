import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tiingo import TiingoClient
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import datetime

# Load artifacts
try:
    model = load_model('stock_prediction_lstm.h5')
    scaler = joblib.load('scaler.save')
except Exception as e:
    st.error(f"Error loading model/scaler: {str(e)}")
    st.stop()

# Streamlit app configuration
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title('📈 Stock Price Prediction App')

# Sidebar inputs
with st.sidebar:
    st.header("User Inputs")
    stock_symbol = st.text_input('Stock Symbol', 'AAPL').upper()
    prediction_days = st.slider('Prediction Days', 7, 30, 30)
    n_steps = 100  # Must match model's training parameter

# Tiingo configuration
config = {
    'session': True,
    'api_key': "40a7cd2dd2cdc63ff01e0d090460b00d47ef771a"
}


def get_stock_data(symbol):
    try:
        client = TiingoClient(config)
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=5 * 365)  # 5 years history
        df = client.get_dataframe(symbol,
                                  startDate=start_date,
                                  endDate=end_date,
                                  frequency='daily')
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


def main():
    # Fetch data with progress indicator
    with st.spinner('Fetching stock data...'):
        df = get_stock_data(stock_symbol)

    if df is None or df.empty:
        st.error("No data available for this stock symbol")
        return

    # Data preprocessing
    with st.spinner('Processing data...'):
        closes = df[['close']].values
        scaled_closes = scaler.transform(closes)
        last_sequence = scaled_closes[-n_steps:].flatten().tolist()

    # Make predictions
    with st.spinner('Generating predictions...'):
        predictions_scaled = []
        current_sequence = last_sequence.copy()

        for _ in range(prediction_days):
            x_input = np.array(current_sequence[-n_steps:]).reshape(1, n_steps, 1)
            yhat = model.predict(x_input, verbose=0)[0][0]
            current_sequence.append(yhat)
            predictions_scaled.append(yhat)

        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

        # Create timeline
        last_date = df.index[-1]
        prediction_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=prediction_days
        )

        # Visualization
        st.subheader(f"Price Predictions for {stock_symbol}")

        # Create figure with Plotly for better interactivity
    try:
        import plotly.graph_objects as go
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=df.index[-n_steps:],
            y=closes[-n_steps:].flatten(),
            name='Historical Prices',
            line=dict(color='blue')
        ))

        # Predictions
        fig.add_trace(go.Scatter(
            x=prediction_dates,
            y=predictions.flatten(),
            name='Predicted Prices',
            line=dict(color='red', dash='dot')
        ))

        fig.update_layout(
            title=f'{stock_symbol} Price Predictions',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback to matplotlib if Plotly not installed
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index[-n_steps:], closes[-n_steps:], label='Historical Prices')
        ax.plot(prediction_dates, predictions, label='Predicted Prices', linestyle='--')
        ax.set_title(f'{stock_symbol} Price Predictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    # Display predictions table
    st.subheader('Prediction Details')
    prediction_df = pd.DataFrame({
        'Date': prediction_dates.date,
        'Predicted Price': predictions.flatten()
    }).set_index('Date')

    st.dataframe(
        prediction_df.style.format({'Predicted Price': '${:.2f}'}),
        use_container_width=True
    )


if __name__ == '__main__':
    main()
