import pandas as pd

# Load the dataset
file_path = "C:\Projects\DL\stock\PSHG_p Historical Data.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Debugging: Print column names to check if 'Close' exists
print("Columns in dataset:", df.columns.tolist())

# Convert 'Date' column to datetime format
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

# Handle missing values in 'Vol.' column
df["Vol."] = df["Vol."].fillna("0")

# Convert 'Vol.' column to numerical values (handle 'K' and 'M' suffixes)
def convert_volume(vol):
    if 'K' in vol:
        return float(vol.replace('K', '')) * 1_000
    elif 'M' in vol:
        return float(vol.replace('M', '')) * 1_000_000
    else:
        return float(vol)

df["Vol."] = df["Vol."].apply(convert_volume)

# Convert 'Change %' column to float (remove '%' and convert to decimal)
df["Change %"] = df["Change %"].str.replace('%', '').astype(float) / 100

# Rename columns for better readability
df.rename(columns={"Vol.": "Volume", "Change %": "Change", "Price": "Close"}, inplace=True)

# Sort data by Date
df = df.sort_values(by="Date")

# Add Moving Averages
df["7_Day_MA"] = df["Close"].rolling(window=7).mean()
df["14_Day_MA"] = df["Close"].rolling(window=14).mean()
df.fillna(0, inplace=True)  # Fill NaN values from rolling mean

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

print("✅ Data cleaned and saved as 'cleaned_data.csv'")


