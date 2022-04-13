import snscrape.modules.twitter as sntwitter
import pandas as pd

query ="Arsenal"
tweets = []
limit = 500000

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    #print(vars(tweet))
    #break
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.username, tweet.content, tweet.likeCount, tweet.replyCount, tweet.retweetCount])

df = pd.DataFrame(tweets, columns =['Date','User','Tweet','TweetLikea','TweetReplies','RetweetCount'])
df.to_csv('./Twitter/Arsenal_Scraper_test.csv')
