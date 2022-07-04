# Import Dictionary
from gensim.corpora.dictionary import Dictionary
from nlp_utils import get_pre_process_wiki_articles

# Create a Dictionary from the articles: dictionary
articles = get_pre_process_wiki_articles()
dictionary = Dictionary(articles)

# Select the id for "computer": computer_id
computer_id = dictionary.token2id.get("computerv")

# Use computer_id with the dictionary to print the word
print(dictionary.get(computer_id))

# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Print the first 10 word ids with their frequency counts from the fifth document
print(corpus[4][:10])