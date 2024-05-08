import pandas as pd

files = ["2353","2357","2515","2890","2891", "4763", "4934", "2888"]
for file in files:
    data = pd.read_csv(f"./datas/{file}.csv")
    data.drop(["Date", "Time", "Timestamp"], axis=1, inplace=True)
    data.rename(columns={"DateTime": "datetime", "Open": "open", "High":"high", "Low":"low", "Close": "close", "Volume": "volume", "Amount": "amount"}, inplace=True)
    data.to_csv(f"./datas/{file}.csv", index=False)