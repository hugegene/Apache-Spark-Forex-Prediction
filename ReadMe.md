Training Module contains 2 sub-modules. The first sub-module, is the Sentiment sub-module, batch process mongoDB news data every end of day to perform sentiment sentiment analysis on it and give each headline news a Sentiment score (0-1) that the price will increase on the next day. The second sub-module, which is the Price sub-module trains a logistic regression model, Price Model, that takes in the Sentiment score, price at t, price at t-1, price at t-2 to predict the price movement at t+1. t here refers to minute.

GetDataFromAPI fetches AlphaAdvantage API every minute on puts on a TCP port. Flume catches the TCP port as a source and push it into a channel.

StreamingModule implement Spark Streaming on every minute basis receive from Flume as a sink. Spark streaming pre-process the mini-batch RDD as a dataframe and predicts with the pre-trained Price Model whether the price movement is up or down on the next time step and execute a buy or short-sell live. The dataframe with the real price, buy/sell execution, returns and cumulative returns are then pushed to MongoDB database for record.

Visualisation takes in MongoDB database and plot the live graph of real price and cumulative profits. The live graph of cumulative profit plot how much a $1 investment turn out over time.

The order of execution of program is:

	Training Module => GetDataFromAPI => Streaming Module =>Visualisation