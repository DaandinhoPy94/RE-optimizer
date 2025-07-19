from utils.data_collector import VastgoedDataCollector

collector = VastgoedDataCollector()
df_historical = collector.get_historical_price_data(years_back=3)
print(df_historical.head())
print(f"\nDataset shape: {df_historical.shape}")
print(f"\nUnieke regio's: {df_historical['regio'].nunique()}")