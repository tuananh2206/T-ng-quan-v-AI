import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

#1
df = pd.read_csv("C:\Data processing\Labs\Lab5\ITA105_Lab_5_Supermarket.csv")
print(df.head())

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
print(df.isnull().sum())
df['revenue_ffill'] = df['revenue'].ffill()
df['revenue_bfill'] = df['revenue'].bfill()
df['revenue_interp'] = df['revenue'].interpolate(method='linear')

df['year'] = df.index.year
df['month'] = df.index.month
df['quarter'] = df.index.quarter

df['day_of_week'] = df.index.dayofweek

df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

monthly_revenue = df['revenue'].resample('ME').sum()
weekly_revenue = df['revenue'].resample('W').sum()

plt.subplot(1, 2, 1) 
weekly_revenue.plot(kind='line', marker='o', color='b')
plt.title('Tổng Doanh Thu Theo Tuần')
plt.xlabel('Thời gian')
plt.ylabel('Doanh thu')

plt.subplot(1, 2, 2) 
monthly_revenue.plot(kind='bar', color='orange')
plt.title('Tổng Doanh Thu Theo Tháng')
plt.xlabel('Tháng')
plt.ylabel('Doanh thu')

plt.tight_layout()
plt.show()

df['7day_MA'] = df['revenue_interp'].rolling(window=7, center=True).mean()
df['30day_MA'] = df['revenue_interp'].rolling(window=30, center=True).mean()

plt.figure(figsize=(12, 6))
plt.plot(df['revenue_interp'], label='Dữ liệu gốc', alpha=0.3, color='gray')
plt.plot(df['7day_MA'], label='Xu hướng tuần (Rolling 7d)', color='blue', linewidth=2)
plt.plot(df['30day_MA'], label='Xu hướng tháng (Rolling 30d)', color='red', linewidth=2)

plt.title('Phát hiện xu hướng dài hạn bằng Rolling Mean')
plt.legend()
plt.show()

result = seasonal_decompose(df['revenue_interp'], model='additive', period=7)
fig = result.plot()
fig.set_size_inches(12, 10)
plt.suptitle('Phân rã Trend & Seasonality bằng Decomposition', fontsize=15)
plt.show()

#=================================================================================
#2
df = pd.read_csv("C:\Data processing\Labs\Lab5\ITA105_Lab_5_Web_traffic.csv")
print(df.head(10))

df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
df_hourly = df.asfreq('h')

missing_count = df_hourly['visits'].isnull().sum()
df_hourly['visits'] = df_hourly['visits'].interpolate(method='linear')

df_hourly['hour'] = df_hourly.index.hour
df_hourly['day_of_week'] = df_hourly.index.dayofweek
df_hourly['day_name'] = df_hourly.index.day_name()

hourly_stats = df_hourly.groupby('hour')['visits'].mean()
plt.figure(figsize=(12, 6))
hourly_stats.plot(kind='line', marker='o', color='teal', linewidth=2, markersize=8)

peak_hour = hourly_stats.idxmax()
peak_value = hourly_stats.max()
trough_hour = hourly_stats.idxmin()
trough_value = hourly_stats.min()

plt.annotate(f'Peak: {peak_hour}h', xy=(peak_hour, peak_value), xytext=(peak_hour+1, peak_value),
             arrowprops=dict(facecolor='red', shrink=0.05))
plt.annotate(f'Trough: {trough_hour}h', xy=(trough_hour, trough_value), xytext=(trough_hour+1, trough_value),
             arrowprops=dict(facecolor='blue', shrink=0.05))

plt.title('Biểu đồ lưu lượng truy cập trung bình theo giờ trong ngày', fontsize=14)
plt.xlabel('Giờ trong ngày (0 - 23h)')
plt.ylabel('Lượt truy cập trung bình (Visits)')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

daily_seasonality = df_hourly.groupby('hour')['visits'].mean()
sns.lineplot(x=daily_seasonality.index, y=daily_seasonality.values, ax=ax1, marker='o', color='teal')
ax1.set_title('Mùa vụ hàng ngày (Lưu lượng trung bình theo Giờ)')
ax1.set_xlabel('Giờ trong ngày')
ax1.set_ylabel('Lượt truy cập trung bình')
ax1.set_xticks(range(24))

weekly_seasonality = df_hourly.groupby('day_of_week')['visits'].mean()
sns.barplot(x=weekly_seasonality.index, y=weekly_seasonality.values, ax=ax2, palette='viridis')
ax2.set_title('Mùa vụ hàng tuần (Lưu lượng trung bình theo Thứ)')
ax2.set_xlabel('Thứ (0=T2, 6=CN)')
ax2.set_ylabel('Lượt truy cập trung bình')
ax2.set_xticklabels(['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN'])
plt.tight_layout()
plt.show()

#====================================================
#3
df = pd.read_csv("C:\Data processing\Labs\Lab5\ITA105_Lab_5_Stock.csv")
print(df.head(10))
print(df.shape)

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

df_reindexed = df.asfreq('B')
print(df_reindexed.isnull().sum())
df_final = df_reindexed.ffill()

plt.figure(figsize=(12, 6))
plt.plot(df_final.index, df_final['close_price'], color='royalblue', linewidth=1.5, label='Giá đóng cửa')
plt.title('Biểu đồ biến động Giá đóng cửa Cổ phiếu (2018 - 2023)', fontsize=14)
plt.xlabel('Thời gian', fontsize=12)
plt.ylabel('Giá (USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

df_final['MA7'] = df_final['close_price'].rolling(window=7).mean()
df_final['MA30'] = df_final['close_price'].rolling(window=30).mean()

plt.figure(figsize=(14, 7))
plt.plot(df_final.index, df_final['close_price'], label='Giá gốc (Daily)', alpha=0.3, color='gray')
plt.plot(df_final.index, df_final['MA7'], label='Trend ngắn hạn (7 ngày)', color='blue', linewidth=1.5)
plt.plot(df_final.index, df_final['MA30'], label='Trend trung hạn (30 ngày)', color='red', linewidth=2)

plt.title('Nhận diện xu hướng Giá cổ phiếu bằng Rolling Mean', fontsize=14)
plt.xlabel('Thời gian')
plt.ylabel('Giá đóng cửa')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

df_final['month'] = df_final.index.month
monthly_seasonality = df_final.groupby('month')['close_price'].mean()

plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_seasonality.index, y=monthly_seasonality.values, palette='coolwarm')

plt.title('Phân tích Mùa vụ: Giá đóng cửa trung bình theo tháng (2018-2023)', fontsize=14)
plt.xlabel('Tháng trong năm')
plt.ylabel('Giá đóng cửa trung bình')
plt.xticks(ticks=range(0, 12), labels=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_final, x='month', y='close_price', palette='Set3')
plt.title('Sự phân bổ giá cổ phiếu theo từng tháng', fontsize=14)
plt.xlabel('Tháng')
plt.ylabel('Giá đóng cửa')
plt.show()

#===================================================================
#4
df = pd.read_csv("C:\Data processing\Labs\Lab5\ITA105_Lab_5_Production.csv")
print(df.head(10))
print(df.shape)

df['week_start'] = pd.to_datetime(df['week_start'])
df.set_index('week_start', inplace=True)
df = df.asfreq('W')
df['production'].isnull().sum()
df['production'] = df['production'].interpolate(method='linear')

df['year'] = df.index.year
df['quarter'] = df.index.quarter
df['week_of_year'] = df.index.isocalendar().week

df['Trend_Quarterly'] = df['production'].rolling(window=12, min_periods=1).mean()
df['Trend_Yearly'] = df['production'].rolling(window=52, min_periods=1).mean()

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['production'], label='Sản lượng hàng tuần', color='gray', alpha=0.3)

plt.plot(df.index, df['Trend_Quarterly'], label='Xu hướng Quý (12-week MA)', color='orange', linewidth=2)
plt.plot(df.index, df['Trend_Yearly'], label='Xu hướng Dài hạn (52-week MA)', color='red', linewidth=3)

plt.title('Phát hiện Xu hướng Dài hạn trong Sản xuất Công nghiệp', fontsize=14)
plt.xlabel('Thời gian')
plt.ylabel('Sản lượng')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

quarterly_seasonality = df.groupby('quarter')['production'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=quarterly_seasonality.index, y=quarterly_seasonality.values, palette='viridis')
plt.title('Mùa vụ theo Quý: Sản lượng trung bình (2020-2022)', fontsize=14)
plt.xlabel('Quý trong năm')
plt.ylabel('Sản lượng trung bình')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Quý 1', 'Quý 2', 'Quý 3', 'Quý 4'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

pivot_df = df.pivot_table(values='production', index='quarter', columns='year', aggfunc='mean')
pivot_df.plot(kind='line', marker='o', figsize=(12, 6))
plt.title('So sánh sản lượng theo Quý giữa các năm', fontsize=14)
plt.xlabel('Quý')
plt.ylabel('Sản lượng')
plt.xticks([1, 2, 3, 4])
plt.legend(title='Năm')
plt.grid(True, alpha=0.3)
plt.show()

result = seasonal_decompose(df['production'], model='additive', period=13)
plt.rcParams['figure.figsize'] = (12, 10)
result.plot()
plt.suptitle('Phân rã chuỗi thời gian Sản lượng Sản xuất (Trend - Seasonality - Residuals)', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()