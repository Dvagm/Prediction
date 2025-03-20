import gradio as gr
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def load_trained_model(model_path='best_lstm_model.keras'):
    return load_model(model_path)

def predict_stock_price(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return "Data tidak tersedia. Coba ticker atau rentang tanggal lain."

        closing_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(closing_prices)

        look_back = 50
        last_look_back = scaled_data[-look_back:]

        model = load_trained_model()
        future_predictions = []
        for _ in range(7):
            input_data = last_look_back.reshape(1, look_back, 1)
            next_pred = model.predict(input_data, verbose=0)
            future_predictions.append(next_pred[0, 0])
            last_look_back = np.append(last_look_back[1:], next_pred, axis=0)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        historical_dates = data.index[-50:]
        historical_prices = closing_prices[-50:].flatten()
        historical_df = pd.DataFrame({'Date': historical_dates, 'Price': historical_prices, 'Type': 'Historical'})

        future_dates = pd.date_range(start=pd.to_datetime(end_date) + pd.Timedelta(days=1), periods=7)
        predicted_df = pd.DataFrame({'Date': future_dates, 'Price': future_predictions.flatten(), 'Type': 'Prediction'})

        combined_df = pd.concat([historical_df, predicted_df])
        fig = px.line(combined_df, x='Date', y='Price', color='Type',
                      title=f"Prediksi Harga Saham {ticker} (7 Hari ke Depan)",
                      markers=True, color_discrete_map={'Historical': 'blue', 'Prediction': 'red'})
        return fig
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"

demo = gr.Interface(
    fn=predict_stock_price,
    inputs=[
        gr.Textbox(label="Ticker Saham", value="AAPL"),
        gr.Textbox(label="Tanggal Mulai (YYYY-MM-DD)", value="2020-01-01"),
        gr.Textbox(label="Tanggal Akhir (YYYY-MM-DD)", value="2025-01-31"),
    ],
    outputs=gr.Plot(label="Grafik Prediksi"),
    title="Prediksi Harga Saham dengan LSTM",
    description="Masukkan ticker saham, tanggal mulai, dan tanggal akhir di butuhkan 5 tahun 30 hari untuk memprediksi harga saham selama 7 hari ke depan.",
)

demo.launch()
