from app import get_news, parse_news, score_news
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report




#df = score_news(parse_news(get_news("BTC")))
#df_polygon = pd.read_csv('2025-06-10_to_2025-06-16_X:BTCUSD.csv')

df = score_news(parse_news(get_news("BTC")))
df = df.reset_index()
df['date'] = df['datetime'].dt.date
daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index()

# Process price data
df_polygon = pd.read_csv('2025-05-29_to_2025-06-30_X:BTCUSD.csv')
df_polygon['Timestamp'] = pd.to_datetime(df_polygon['Timestamp'])
df_polygon['date'] = df_polygon['Timestamp'].dt.date

# Merge
merged_df = df_polygon.merge(daily_sentiment, on='date', how='left')
merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(0)



merged_df = merged_df.sort_values('date')

merged_df['delta_close'] = merged_df['Close'].diff()

merged_df['sentiment_correct'] = np.where(
    np.sign(merged_df['sentiment_score']) == np.sign(merged_df['delta_close']),
    1,
    0
)

# Make sure data is sorted
merged_df = merged_df.sort_values('date')

# Compute binary label: was sentiment correct for today's price change?
merged_df['sentiment_correct'] = np.where(
    np.sign(merged_df['sentiment_score']) == np.sign(merged_df['Close'].diff()),
    1,
    0
)

# Compute binary label: was sentiment correct for *tomorrow's* price change?
merged_df['sentiment_correct_next_day'] = np.where(
    np.sign(merged_df['sentiment_score']) == np.sign(merged_df['Close'].shift(-1) - merged_df['Close']),
    1,
    0
)

merged_df['rolling_sentiment_3d'] = merged_df['sentiment_score'].rolling(window=3, min_periods=1).mean()
merged_df['rolling_sentiment_5d'] = merged_df['sentiment_score'].rolling(window=5, min_periods=1).mean()
merged_df['return_1d'] = merged_df['Close'].pct_change()
merged_df['return_3d'] = merged_df['Close'].pct_change(periods=3)
merged_df['sentiment_lag_1d'] = merged_df['sentiment_score'].shift(1)
merged_df['sentiment_lag_2d'] = merged_df['sentiment_score'].shift(2)

merged_df['return_1d_lag'] = merged_df['return_1d'].shift(1)
merged_df['return_3d_lag'] = merged_df['return_3d'].shift(1)

merged_df['high_low_range'] = merged_df['High'] - merged_df['Low']
merged_df['close_open_range'] = merged_df['Close'] - merged_df['Open']

merged_df['sentiment_x_volume'] = merged_df['sentiment_score'] * merged_df['Volume']
merged_df['sentiment_x_return'] = merged_df['sentiment_score'] * merged_df['return_1d']


merged_df['rolling_volatility_3d'] = merged_df['return_1d'].rolling(window=3).std()

features = merged_df[[
    'sentiment_score',
    'Close',
    'Open',
    'High',
    'Low',
    'Volume',
    'Transactions',
    'sentiment_correct',
    'sentiment_correct_next_day',
    'rolling_sentiment_3d',
    'rolling_sentiment_5d',
    'return_1d',
    'return_3d',
    'sentiment_lag_1d',
    'sentiment_lag_2d',
    'return_1d_lag',
    'return_3d_lag',
    'high_low_range',
    'close_open_range',
    'sentiment_x_volume',
    'sentiment_x_return',
    'rolling_volatility_3d'
    ]]

# Optional: convert the target to categorical for hue
#features['sentiment_score'] = features['sentiment_score'].astype(str)

# Create pairplot (scatter matrix) colored by sentiment accuracy
sns.pairplot(features, corner=False, plot_kws={'alpha': 0.6})

plt.suptitle('Scatter Matrix: Price, Sentiment, and Sentiment Accuracy', y=1.02)
plt.tight_layout()
plt.savefig("scatter_matrix.png", dpi=300, bbox_inches='tight')

print(merged_df)

ft = merged_df.drop(columns=['Timestamp', 'sentiment_correct', 'date'])
X = ft
y = merged_df['sentiment_correct']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=1000,
    max_depth=2,
    learning_rate=0.2
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# save dataset to CSV
merged_df.to_csv('merged_btc_sentiment_data.csv', index=False)