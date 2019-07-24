# TwitterPred

A tool to predict future user behavior on twitter.

## Features

Uses machine learning to predict how a given user may respond to a given set of tweets. 
Such responses include:
 - retweet
 - reply
 - like
 - quote
 - no action (the user chooses not to respond to the tweet)

## Usage

Install the requirements.txt file.
Provide Two Files called tweet.csv and user.csv.

 - user.csv: contains your twitter developer credentials and the twitter username of the user you may wish to predict.
   - Example:
   
      ``` 
      consumer_key,DHSAJHSAJHJHS21
      consumer_secret,DFJHKFJKDSSFJKDJFDJH21
      access_key,XVCJHFJAJHSFDH2
      access_secret,GJHKSDJHFDJHBH2
      user,realDonaldTrump
      ```

- tweet.csv: contains a list of twitter IDs that you want to check the probability of how the given user may respond to. 
  - Example: 
  
      ``` 
      ID,
      73478273823891,
      621873981209091,
      ```
Run the script twitterPred.py.

## Results

After the script finish executing, the outputs would be stored in (username)_tweets_probability.csv

 - Example: 
  
      ``` 
      ID,retweet,reply,like,quote,no action
      73478273823891,0.1221,0.21213,0.21782187,0.213321,0.319
      621873981209091,0.2123,0.1257,0.12743,0.21146,0.3213
      ```
