import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import feedparser
import time
from datetime import datetime
# import pickle

## TODO - clean up code

# create function to get the subjectivity
def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

# create function to get the polarity
def getPolarity(text):
  return TextBlob(text).sentiment.polarity

# create function to get the sentiment scores
def getSIA(text):
  # returns a dictionary of scores (neg, neu, pos, compound)
  sia = SentimentIntensityAnalyzer()
  sentiment = sia.polarity_scores(text)
  return sentiment

# Yahoo Finance News
# TODO Add more news sources
# TODO Add historic news data
def fetchNews(ticker):
  # get news from yahoo finance
  feedurl = 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=%s&region=US&lang=en-US'%ticker
  # parse feed
  feed = feedparser.parse(feedurl)
  # if no articles found, skip
  if len(feed.entries) == 0: print("No articles found for "+ticker+"...")
  # format data for dataframe
  else:
    rows = []
    for entry in feed.entries:
      # format date
      formatted_date = str(entry.published_parsed.tm_year)+'-'+str(entry.published_parsed.tm_mon).zfill(2)+'-'+str(entry.published_parsed.tm_mday).zfill(2)
      # construct the data dictionary
      data = { 'Ticker': ticker, 'Date': formatted_date, 'Article': entry.title+' - '+entry.summary }
      # append the data dictionary to the list
      rows.append(data)

    # convert to dataframe
    df1 = pd.DataFrame(rows)
    # subjectivity - how subjective or opinionated the text is (0 = fact, 1 = opinion)
    df1['Subjectivity'] = df1['Article'].apply(getSubjectivity)
    # polarity - how positive or negative the text is (-1 = negative, 1 = positive)
    df1['Polarity'] = df1['Article'].apply(getPolarity)
    # compound - normalized weighted composite score (higher = more positive, lower = more negative)
    df1['Compound'] = df1['Article'].apply(getSIA).apply(lambda x: x['compound'])
    # negative - normalized weighted composite score (higher = more negative)
    df1['Negative'] = df1['Article'].apply(getSIA).apply(lambda x: x['neg'])
    # neutral - normalized weighted composite score (higher = more neutral)
    df1['Neutral'] = df1['Article'].apply(getSIA).apply(lambda x: x['neu'])
    # positive - normalized weighted composite score (higher = more positive)
    df1['Positive'] = df1['Article'].apply(getSIA).apply(lambda x: x['pos'])
    # save to csv
    df1.to_csv('./news/%s.csv'%ticker)
    return df1

def fetchStockPrices(ticker):
  print('Downloading historical price data for '+ticker+'...')
  print("")
  data = yf.download(tickers, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
  data['Ticker'] = ticker
  data['Label'] = np.where(data['Close'].shift(-1) > data['Close'], 0, 1)
  data.to_csv('./prices/%s.csv'%ticker)
  return data

#######################################################################

# select the tickers you want
tickers = ['AMZN']
# other_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB', 'BRK-A', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'DIS', 'PYPL', 'CMCSA', 'VZ', 'NFLX', 'ADBE', 'KO', 'XOM', 'PEP', 'T', 'INTC', 'PFE', 'MRK', 'CSCO', 'CVX', 'WFC', 'ABT', 'CRM', 'BMY', 'ABBV', 'DHR', 'MCD', 'AMGN', 'ACN', 'AVGO', 'TMO', 'MDT', 'COST', 'NEE', 'TXN', 'UNP', 'NKE', 'HON', 'LLY', 'PM', 'LIN', 'UPS', 'SBUX', 'ORCL', 'LOW', 'IBM', 'AMD', 'QCOM', 'CAT', 'BA', 'GS', 'MMM', 'GE', 'AMT']

# loop through tickers and fetch news and stock prices
for ticker in tickers:
  print('Training model for '+ticker)
  newsdf = fetchNews(ticker)
  # TODO check there's data in the dataframe
  pricedf = fetchStockPrices(ticker)
  # get the current average sentiment based on all news fetched
  averageSentiment = newsdf.groupby('Ticker').mean()
  print(averageSentiment)
  # merge the dataframes
  df = pd.merge(pricedf, averageSentiment, on='Ticker')
  # TODO - get sentiment data to historical data training
  # ignore sentiment data for historical data training
  df = df.drop(['Ticker', 'Subjectivity', 'Polarity', 'Compound', 'Negative', 'Neutral', 'Positive'], axis=1)
  # remove the Ticker column
  # df = df.drop(['Ticker'], axis=1)
  df.to_csv('./prices/%s.csv'%ticker)
  # create the feature dataset
  X = df
  X = np.array(X.drop(['Label'], axis=1))
  # create the target dataset
  y = np.array(df['Label'])
  # split the data into 80% training and 20% testing datasets
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
  # create and train the model
  model = LinearDiscriminantAnalysis().fit(x_train, y_train)
  # get the models predictions and classifications
  predictions = model.predict(x_test)
  # show the models metrics
  accuracy = accuracy_score(y_test, predictions)
  print('-----------------------------------------------------')
  print('Model Accuracy: ', (accuracy * 100).round(2), '%')
  print('-----------------------------------------------------')
  # print('Classification report: ', classification_report(y_test, predictions))
  # save the model for later use
  # pickle.dump(model, open('model.sav', 'wb'))
  # load the model
  # loaded_model = pickle.load(open('model.sav', 'rb'))
  # make a prediction using the loaded model
  # result = loaded_model.score(x_test, y_test)
  # print(result)
  # print('Accuracy score: ', accuracy_score(y_test, predictions))
  # print('Classification report: ', classification_report(y_test, predictions))
  # print('Predictions: ', predictions)