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

    # Get the past 6 months of data instead of just 7 days
    # Assuming the data is sorted by date, get the last 180 days (approximately 6 months)
    past_6_months = df['Date'].tail(180).tolist()
    past_prices = df['Close'].tail(180).tolist()

    # Fetch latest stock price
    stock_data = get_stock_data("AAPL")
    latest_price = stock_data["close"] if stock_data else past_prices[-1]

    # Feature input for prediction
    st.sidebar.header("Stock Price Prediction")
    st.sidebar.write(f"📌 Latest Real-Time Price: ${latest_price:.2f}")

    # For prediction, we still need the last 7 days of data
    prediction_data = past_prices[-7:]
    
    # Ensure we have exactly 7 data points for prediction
    if len(prediction_data) > 7:
        prediction_data = prediction_data[-7:]
    elif len(prediction_data) < 7:
        # Pad with the earliest available data if we have less than 7 points
        padding = [prediction_data[0]] * (7 - len(prediction_data))
        prediction_data = padding + prediction_data

    # Predict next day's price
    if st.sidebar.button("Predict Tomorrow's Price"):
        scaled_input = scaler.transform([prediction_data])
        predicted_price = model.predict(scaled_input)[0]
        st.sidebar.success(f"📊 Predicted Price: {predicted_price:.2f}")
        
        # Add the predicted price to our display data
        tomorrow_date = "Tomorrow"
        past_6_months.append(tomorrow_date)
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

    # Display recent stock records (keep showing just the last 7 days in the table)
    st.subheader("📊 Ravi Teja's Porsche Stock Performance")
    st.write(df[['Date', 'Open', 'Close', 'Change']].tail(7))

    # Display user's holdings dynamically
    st.subheader("📌 User Holdings")
    st.write(f"Stocks Owned: {st.session_state.num_stocks}")
    st.write(f"Total Investment: ${st.session_state.num_stocks * latest_price:.2f}")
    st.write(f"Available Balance: ${st.session_state.balance:.2f}")

    # Create interactive stock trend chart with 6 months of data
    # For better visualization of 6 months of data, we'll show fewer x-axis labels
    # by selecting every 15th day (approximately 2 weeks)
    display_dates = past_6_months[::15]  # Take every 15th date
    display_indices = list(range(0, len(past_6_months), 15))
    
    # Make sure we include the last point (and tomorrow's prediction if it exists)
    if len(past_6_months) - 1 not in display_indices:
        display_dates.append(past_6_months[-1])
        display_indices.append(len(past_6_months) - 1)
    
    # Create x-axis labels with reduced frequency
    x_axis_labels = [""] * len(past_6_months)
    for i, date in zip(display_indices, display_dates):
        x_axis_labels[i] = date

    line = (
        Line()
        .add_xaxis(past_6_months)
        .add_yaxis("Stock Price", past_prices, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Ravi Teja Stock Price Trend (6 Months)"),
            xaxis_opts=opts.AxisOpts(
                type_="category", 
                boundary_gap=False,
                axislabel_opts=opts.LabelOpts(rotate=45, interval=15)  # Rotate labels and show every 15th label
            ),
            yaxis_opts=opts.AxisOpts(type_="value"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),  # Show tooltip on hover
            datazoom_opts=[
                opts.DataZoomOpts(range_start=0, range_end=100),  # Add zoom functionality
                opts.DataZoomOpts(type_="inside")
            ],
        )
    )

    # Display chart
    st_pyecharts(line)