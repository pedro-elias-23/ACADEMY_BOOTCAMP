# Import the necessary modules
from nltk.tokenize import regexp_tokenize, TweetTokenizer
from nlp_utils import get_tweets_sample

tweets = get_tweets_sample()

# Define a regex pattern to find hashtags: pattern1
pattern1 = r"#\w+"
# Use the pattern on the first tweet in the tweets list
hashtags = tweets[0]
print(regexp_tokenize(hashtags, pattern1))

# Write a pattern that matches both mentions (@) and hashtags
pattern2 = r"(@\w+|#\w)"
# Use the pattern on the last tweet in the tweets list
mentions_hashtags = tweets[-1]
print(regexp_tokenize(mentions_hashtags, pattern2))

# Use the TweetTokenizer to tokenize all tweets into one list
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets]
print(all_tokens)