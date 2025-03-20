import streamlit as st
import joblib
import numpy as np
import datetime
import pandas as pd
import requests
from pyecharts.charts import Line
from pyecharts import options as opts
from streamlit_echarts import st_pyecharts
from datetime import datetime as dt

# Load trained model and scaler
model = joblib.load("stock_price_model_new.pkl")
scaler = joblib.load("stock_price_scaler_new.pkl")

# Streamlit UI
st.set_page_config(page_title="Ravi Teja's Porsche Stock Predictor", layout="wide")
st.title("📈 Ravi Teja's Porsche Stock Predictor")

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
        st.rerun()

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
    
    # Convert Date column to datetime for better date handling
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except:
        # If direct conversion fails, try with a specific format
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        # Drop any rows with invalid dates
        df = df.dropna(subset=['Date'])
    
    # Sort dataframe by date
    df = df.sort_values('Date')
    
    # Get min and max dates for the date picker
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    # Create date range presets
    st.header("📅 Select Date Range")
    
    # Create preset options
    date_presets = {
        "Last Month": (max_date - pd.Timedelta(days=30), max_date),
        "Last 3 Months": (max_date - pd.Timedelta(days=90), max_date),
        "Last 6 Months": (max_date - pd.Timedelta(days=180), max_date),
        "Last Year": (max_date - pd.Timedelta(days=365), max_date),
        "Year to Date": (datetime.date(max_date.year, 1, 1), max_date),
        "All Time": (min_date, max_date),
        "Custom": (None, None)
    }
    
    # Create columns for range selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_preset = st.selectbox("Select Range", list(date_presets.keys()), index=3)  # Default to Last Year
    
    # Set dates based on preset or allow custom input
    if selected_preset != "Custom":
        start_date, end_date = date_presets[selected_preset]
    else:
        with col2:
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start Date", 
                                          value=max_date - pd.Timedelta(days=365),
                                          min_value=min_date,
                                          max_value=max_date)
            with date_col2:
                end_date = st.date_input("End Date", 
                                        value=max_date,
                                        min_value=min_date,
                                        max_value=max_date)
    
    # Filter data based on selected date range
    mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
    filtered_df = df.loc[mask]
    
    # Check if we have data in the selected range
    if filtered_df.empty:
        st.error("No data available for the selected date range. Please select a different range.")
    else:
        # Convert dates to string format for chart
        date_strings = filtered_df['Date'].dt.strftime('%Y-%m-%d').tolist()
        prices = filtered_df['Close'].tolist()
        
        # Fetch latest stock price
        stock_data = get_stock_data("AAPL")
        latest_price = stock_data["close"] if stock_data else prices[-1]
        
        # Feature input for prediction
        st.sidebar.header("Stock Price Prediction")
        st.sidebar.write(f"📌 Latest Real-Time Price: ${latest_price:.2f}")
        
        # For prediction, we need the last 7 days of data
        prediction_data = df['Close'].tail(7).tolist()
        
        # Ensure we have exactly 7 data points for prediction
        if len(prediction_data) > 7:
            prediction_data = prediction_data[-7:]
        elif len(prediction_data) < 7:
            padding = [prediction_data[0]] * (7 - len(prediction_data))
            prediction_data = padding + prediction_data
        
        # Predict next day's price
        if st.sidebar.button("Predict Tomorrow's Price"):
            scaled_input = scaler.transform([prediction_data])
            predicted_price = model.predict(scaled_input)[0]
            st.sidebar.success(f"📊 Predicted Price: ${predicted_price:.2f}")
            
            # Add predicted price to display data if today's date is in the filtered data
            if max_date == df['Date'].max().date():
                tomorrow_date = "Tomorrow"
                date_strings.append(tomorrow_date)
                prices.append(predicted_price)
        
        # Buy/Sell Section
        st.sidebar.header("💰 Buy/Sell Stocks")
        st.sidebar.write(f"💰 Current Balance: ${st.session_state.balance:.2f}")
        st.sidebar.write(f"📈 Stocks Owned: {st.session_state.num_stocks}")
        num_stocks = st.sidebar.number_input("Enter Number of Stocks to Buy/Sell", min_value=1, value=1)
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
        
        # Display recent stock records
        st.subheader("📊 Recent Stock Performance")
        st.write(filtered_df[['Date', 'Open', 'Close', 'Change']].tail(7))
        
        # Display user's holdings dynamically
        st.subheader("📌 User Holdings")
        st.write(f"Stocks Owned: {st.session_state.num_stocks}")
        st.write(f"Total Investment: ${st.session_state.num_stocks * latest_price:.2f}")
        st.write(f"Available Balance: ${st.session_state.balance:.2f}")
        
        # Profit/Loss Calculator
        st.subheader("💹 Profit/Loss Calculator")
        calc_col1, calc_col2 = st.columns(2)
        
        with calc_col1:
            # Get all dates in the filtered range for the selectors
            available_dates = filtered_df['Date'].dt.strftime('%Y-%m-%d').tolist()
            
            # Create indices for selection
            date_options = {str(date): i for i, date in enumerate(available_dates)}
            
            # Create select boxes with search functionality
            entry_date_index = st.selectbox("Entry Date", 
                                         options=list(date_options.keys()),
                                         index=0,
                                         format_func=lambda x: x)
            
            # Get the price on entry date
            entry_index = date_options[entry_date_index]
            entry_price = filtered_df.iloc[entry_index]['Close']
            st.write(f"Entry Price: ${entry_price:.2f}")
        
        with calc_col2:
            # Default to last date in range
            exit_date_index = st.selectbox("Exit Date", 
                                         options=list(date_options.keys()),
                                         index=len(date_options)-1,
                                         format_func=lambda x: x)
            
            # Get the price on exit date
            exit_index = date_options[exit_date_index]
            exit_price = filtered_df.iloc[exit_index]['Close']
            st.write(f"Exit Price: ${exit_price:.2f}")
        
        # Calculate profit/loss
        hypothetical_shares = st.number_input("Number of Shares for Calculation", min_value=1, value=10)
        
        profit_loss = (exit_price - entry_price) * hypothetical_shares
        profit_loss_percentage = ((exit_price - entry_price) / entry_price) * 100
        
        # Display the profit/loss
        if profit_loss >= 0:
            st.success(f"Profit: ${profit_loss:.2f} ({profit_loss_percentage:.2f}%)")
        else:
            st.error(f"Loss: ${profit_loss:.2f} ({profit_loss_percentage:.2f}%)")
        
        # Investment summary
        investment_amount = entry_price * hypothetical_shares
        final_value = exit_price * hypothetical_shares
        
        st.write(f"Initial Investment: ${investment_amount:.2f}")
        st.write(f"Final Value: ${final_value:.2f}")
        
        # Create a container with specified height for the chart
        chart_container = st.container()
        
        with chart_container:
            # Determine interval based on data size
            num_points = len(date_strings)
            if num_points > 200:
                interval = int(num_points / 20)  # Show approximately 20 labels
            else:
                interval = max(1, int(num_points / 10))  # Show approximately 10 labels
            
            # Mark the entry and exit points
            mark_points = []
            if entry_index <= len(prices) - 1:
                mark_points.append(
                    opts.MarkPointItem(
                        name="Entry",
                        coord=[entry_index, prices[entry_index]],
                        symbol="pin",
                        value=f"Entry: ${prices[entry_index]:.2f}"
                    )
                )
            
            if exit_index <= len(prices) - 1:
                mark_points.append(
                    opts.MarkPointItem(
                        name="Exit",
                        coord=[exit_index, prices[exit_index]],
                        symbol="pin",
                        value=f"Exit: ${prices[exit_index]:.2f}"
                    )
                )
            
            # Create line chart
            line = (
                Line()
                .add_xaxis(date_strings)
                .add_yaxis(
                    "Stock Price", 
                    prices, 
                    is_smooth=True, 
                    label_opts=opts.LabelOpts(is_show=False),
                    markpoint_opts=opts.MarkPointOpts(data=mark_points)
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="Ravi Teja's Porsche Stock Price Trend"),
                    xaxis_opts=opts.AxisOpts(
                        type_="category", 
                        boundary_gap=False,
                        axislabel_opts=opts.LabelOpts(rotate=45, interval=interval)
                    ),
                    yaxis_opts=opts.AxisOpts(
                        type_="value",
                        name="Price ($)",
                        name_location="middle",
                        name_gap=40
                    ),
                    legend_opts=opts.LegendOpts(pos_top="5%"),
                    tooltip_opts=opts.TooltipOpts(
                        trigger="axis",
                        axis_pointer_type="cross",
                        formatter="{b}: ${c}"
                    ),
                    datazoom_opts=[
                        opts.DataZoomOpts(range_start=0, range_end=100),
                        opts.DataZoomOpts(type_="inside")
                    ],
                    toolbox_opts=opts.ToolboxOpts(
                        feature={
                            "saveAsImage": {},
                            "restore": {},
                            "dataZoom": {},
                            "dataView": {"readOnly": False}
                        }
                    ),
                )
                .set_series_opts(
                    linestyle_opts=opts.LineStyleOpts(width=2),
                    itemstyle_opts=opts.ItemStyleOpts(border_width=0),
                )
            )
            
            # Display chart
            st_pyecharts(
                line,
                height="600px",
                width="100%"
            )