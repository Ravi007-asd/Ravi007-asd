import streamlit as st
import joblib
import numpy as np
import datetime
import pandas as pd
import requests
from pyecharts.charts import Line
from pyecharts import options as opts
from streamlit_echarts import st_pyecharts

# Load trained model and scaler
model = joblib.load("stock_price_model_new.pkl")
scaler = joblib.load("stock_price_scaler_new.pkl")

# Streamlit UI
st.set_page_config(page_title="Porsche Stock Predictor", layout="wide")
st.title("📈 Porsche Stock Predictor")

# Initialize session state for user holdings if not already set
if 'balance' not in st.session_state:
    st.session_state.balance = 10000  # Default balance that will be overridden
if 'num_stocks' not in st.session_state:
    st.session_state.num_stocks = 0
if 'balance_set' not in st.session_state:
    st.session_state.balance_set = False

# Let user choose initial balance if not already set
if not st.session_state.balance_set:
    st.header("💰 Set Your Initial Balance")
    initial_balance = st.number_input("Enter your initial balance ($)", min_value=100.0, max_value=1000000.0, value=10000.0, step=100.0)
    if st.button("Set Balance"):
        st.session_state.balance = initial_balance
        st.session_state.balance_set = True
        st.success(f"Initial balance set to ${initial_balance:.2f}")
        st.rerun()  # Using st.rerun() instead of st.experimental_rerun()

# Only show the main app if balance is set
if st.session_state.balance_set:
    # Fetch real-time stock data
    def get_stock_data(symbol="AAPL"):
        api_key = "your_actual_api_key"  # Replace with your API Key
        url = f"https://api.polygon.io/v1/open-close/{symbol}/latest?adjusted=true&apiKey={api_key}"
        try:
            response = requests.get(url).json()
            if "status" in response and response["status"] == "OK" and "close" in response:
                return response
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
        return None

    # Load cleaned stock data
    df = pd.read_csv("cleaned_data.csv")

    # Simulated stock data for past 7 days (Replace with actual logic)
    past_7_days = df['Date'].tail(7).tolist()
    past_prices = df['Close'].tail(7).tolist()

    # Fetch latest stock price
    stock_data = get_stock_data("AAPL")
    latest_price = stock_data["close"] if stock_data else past_prices[-1]

    # Feature input for prediction
    st.sidebar.header("Stock Price Prediction")
    st.sidebar.write(f"📌 Latest Real-Time Price: ${latest_price:.2f}")

    # Ensure exactly 7 past prices, including today's input
    if len(past_prices) == 7:
        past_prices.pop(0)
    past_prices.append(latest_price)

    # Predict next day's price
    if st.sidebar.button("Predict Tomorrow's Price"):
        scaled_input = scaler.transform([past_prices])
        predicted_price = model.predict(scaled_input)[0]
        st.sidebar.success(f"📊 Predicted Price: {predicted_price:.2f}")
        past_7_days.append("Tomorrow")
        past_prices.append(predicted_price)

    # Buy/Sell Section
    st.sidebar.header("💰 Buy/Sell Stocks")
    st.sidebar.write(f"💰 Current Balance: ${st.session_state.balance:.2f}")
    st.sidebar.write(f"📈 Stocks Owned: {st.session_state.num_stocks}")
    num_stocks = st.sidebar.number_input("Enter Number of Stocks to Buy", min_value=1, value=1)
    total_cost = latest_price * num_stocks

    if st.sidebar.button("Buy"):
        if st.session_state.balance >= total_cost:
            st.session_state.balance -= total_cost
            st.session_state.num_stocks += num_stocks
            st.sidebar.success(f"✅ Purchased {num_stocks} stocks for ${total_cost:.2f}")
        else:
            st.sidebar.error("❌ Insufficient balance!")

    if st.sidebar.button("Sell"):
        if num_stocks <= st.session_state.num_stocks:
            total_sale = latest_price * num_stocks
            st.session_state.balance += total_sale
            st.session_state.num_stocks -= num_stocks
            st.sidebar.success(f"📉 Sold {num_stocks} stocks at ${latest_price:.2f} each")
        else:
            st.sidebar.error("❌ Not enough stocks to sell!")

    # Display stock records
    st.subheader("📊 Ravi Teja's Porsche Stock Performance")
    st.write(df[['Date', 'Open', 'Close', 'Change']].tail(7))

    # Display user's holdings dynamically
    st.subheader("📌 User Holdings")
    st.write(f"Stocks Owned: {st.session_state.num_stocks}")
    st.write(f"Total Investment: ${st.session_state.num_stocks * latest_price:.2f}")
    st.write(f"Available Balance: ${st.session_state.balance:.2f}")

    # Create interactive stock trend chart
    line = (
        Line()
        .add_xaxis(past_7_days)
        .add_yaxis("Stock Price", past_prices, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Ravi Teja Stock Price Trend"),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
            yaxis_opts=opts.AxisOpts(type_="value"),
        )
    )

    # Display chart
    st_pyecharts(line)