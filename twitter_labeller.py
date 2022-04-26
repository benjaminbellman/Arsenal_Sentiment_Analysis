'''Import Packages'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd

'''Get Our tweet'''
tweet1 ="@MehranShakarami This team will be the death of me. Absolutely pathetic display. Team is in shambles 😒 https://mehranshakarami.com"
tweet2 ="@MehranShakarami Why do we keep losing!!!"
tweet3 ="WE ARE THE ARSENAL! what an amazing display by the team!!! So proud of the boys."

tweet4 ="I think we need to stay positive, clearly there are improvements and we should look forward to brighter days"
tweet5 ="Team didn't do well, but we will bounce back #comeongunnners"
tweet6 ="Worst team I've seen, pathetic display, no courage 😢"

tweet7 ="I'm so happy Arsenal are back to where they should be"
tweet8 ="AMAZING TEAM DISPLAY, we're back!! WOOHOOO "
tweet9 ="Decent performance, I still think Lacazette can be playing a bit better but very happy about the boy's performance."



'''Preprocess Tweet '''
all_tweets = [tweet1, tweet2, tweet3,tweet4,tweet5,tweet6,tweet7,tweet8, tweet9]
tweet_words =[]
tweets_scores_list = []
for tweet in all_tweets: 
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word ='@user'

        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)
    #print(tweet_words)
    #print(tweet_proc)

    '''Load the model and tokenizer'''
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    


    '''Sentiment analysis''' 
    encoded_tweet = tokenizer(tweet_proc, return_tensors = 'pt')
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    #print(scores)

    tweets_scores_list .append(scores)

labels =['Negative','Neutral','Positive']
tweets_scores = pd.DataFrame(tweets_scores_list, columns = labels)
tweets_scores['Max'] = tweets_scores.idxmax(axis=1)
tweets_scores.head(10)