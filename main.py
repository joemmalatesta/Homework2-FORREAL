# Yeah
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from datasets import load_dataset
# dataset = load_dataset("wikipedia", "20220301.simple")

#######################

# First run. Download datasets of 3000 wikipedia articles from huggingface. normalized and train models and save as word2vec files

#######################
# nltk.download('punkt')
# nltk.download('stopwords')

# # Normalize text from nltk and chatGPT
# def normalizeText(text):
#     return [word.lower() for word in word_tokenize(text) if word.lower() not in stopwords.words('english') and word not in string.punctuation]

# normalized_texts = [normalizeText(doc) for doc in dataset['train'][:3000]['text']]

# # Then, use `normalized_texts` for model training
# model_sg = Word2Vec(sentences=normalized_texts, vector_size=100, window=5, min_count=5, sg=1)
# model_cbow = Word2Vec(sentences=normalized_texts, vector_size=100, window=5, min_count=5, sg=0)

# # Save the model
# model_sg.save("model_sg.word2vec")
# model_cbow.save("model_cbow.word2vec")





















#######################

#After models have been loaded, trained, and saved locally (3000 articles)
# Compare models and outputs in a few lexical challenges

#######################

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# # Load the CBOW and Skip-gram models
cbowModel = Word2Vec.load("model_cbow.word2vec")
sgModel = Word2Vec.load("model_sg.word2vec")

# # Load the Google News Word2Vec model
# googleNewsModel = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

# # Load the FastText model
# fastTextModel = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', binary=False)

# # Define your test cases
# words = ["piano", "gay", "moon"]
# for word in words:
#     # CBOW
#     similarWordsCbow = cbowModel.wv.most_similar(word, topn=10)
#     # Skip-gram
#     similarWordsSg = sgModel.wv.most_similar(word, topn=10)
#     # Google News
#     similarWordsGoogleNews = googleNewsModel.most_similar(word, topn=10)
#     # FastText
#     similarWordsFastText = fastTextModel.most_similar(word, topn=10)

#     print(f"Words similar to '{word}':")
#     print(f"CBOW: {similarWordsCbow}")
#     print(f"SG: {similarWordsSg}")
#     print(f"Google News: {similarWordsGoogleNews}")
#     print(f"FastText: {similarWordsFastText}\n")

# # Analogy test
# analogyCbow = cbowModel.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# analogySg = sgModel.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# analogyGoogleNews = googleNewsModel.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# analogyFastText = fastTextModel.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)

# print(f"Woman + King - Man analogy:")
# print(f"CBOW: {analogyCbow}")
# print(f"SG: {analogySg}")
# print(f"Google News: {analogyGoogleNews}")
# print(f"FastText: {analogyFastText}\n")

# # Similarity test
# similarityCbow = cbowModel.wv.similarity('woman', 'man')
# similaritySg = sgModel.wv.similarity('woman', 'man')
# similarityGoogleNews = googleNewsModel.similarity('woman', 'man')
# similarityFastText = fastTextModel.similarity('woman', 'man')

# print(f"Man/Woman similarity:")
# print(f"CBOW: {similarityCbow}")
# print(f"SG: {similaritySg}")
# print(f"Google News: {similarityGoogleNews}")
# print(f"FastText: {similarityFastText}\n")
























#######################

# Bias in word embeddings

#######################


# # Bias in word embeddings - WEAT from WEFE
# from wefe.datasets import load_bingliu
# from wefe.metrics import RNSB
# from wefe.query import Query
# from wefe.word_embedding_model import WordEmbeddingModel

# import pandas as pd
# import plotly.express as px
# import gensim.downloader as api

# # load the target word sets.
# # In this case each word is an objective set because each of them represents a different social group.
# ethnicity_nationality_words = [
#     ["african", "asian", "hispanic", "native"],  # Groups typically considered minorities
#     ["european", "american", "western"],  # Groups often associated with the majority or dominant culture
# ]

# religion_related_words = [
#     ["christian", "islam", "judaism", "buddhism", "hinduism"],
#     ["atheist", "agnostic", "secular"],
# ]

# bing_liu = load_bingliu()

# # Create the query
# query = Query(religion_related_words, [bing_liu["positive_words"], bing_liu["negative_words"]])



# def evaluate(
#     query: Query, gensim_model_name: str, short_model_name: str, model_args: dict = {}
# ):
#     # Fetch the model
#     model = WordEmbeddingModel(
#         api.load(gensim_model_name), short_model_name, **model_args
#     )

#     # Run the queries
#     results = RNSB().run_query(
#         query, model, holdout=True, print_model_evaluation=True, n_iterations=100
#     )

#     # Show the results obtained with glove
#     fig = px.bar(
#         pd.DataFrame(
#             results["negative_sentiment_distribution"].items(),
#             columns=["Word", "Sentiment distribution"],
#         ),
#         x="Word",
#         y="Sentiment distribution",
#         title=f"{short_model_name} Negative Sentiment Distribution",
#     )

#     fig.update_yaxes(range=[0, 0.2])
#     fig.show()


# evaluate(
#     query,
#     "conceptnet-numberbatch-17-06-300",
#     "Conceptnet",
#     model_args={"vocab_prefix": "/c/en/"},
# )



# def evaluate_with_preloaded_model(query, preloaded_model, short_model_name):
#     # Wrap the pre-loaded model with WEFE's WordEmbeddingModel
#     model = WordEmbeddingModel(preloaded_model.wv, short_model_name)  # Use .wv to get KeyedVectors

#     # Run the queries
#     results = RNSB().run_query(
#         query, model, holdout=True, print_model_evaluation=True, n_iterations=100
#     )

#     # Show the results obtained with the pre-loaded model
#     fig = px.bar(
#         pd.DataFrame(
#             results["negative_sentiment_distribution"].items(),
#             columns=["Word", "Sentiment distribution"],
#         ),
#         x="Word",
#         y="Sentiment distribution",
#         title=f"{short_model_name} Negative Sentiment Distribution",
#     )

#     fig.update_yaxes(range=[0, 0.2])
#     fig.show()


# def check_coverage(model, words):
#     not_found = {word: model.wv.has_index_for(word) for word in words}
#     not_found = {word: has_index for word, has_index in not_found.items() if not has_index}
#     return not_found




















#######################

# CLASSIFICATION ^^^^ Load models with code above for use here

#######################
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# BOW DATASET
dataset = load_dataset("imdb", split='train')
dataset = dataset.shuffle(seed=42).select(range(1000))  # Using a subset for efficiency
trainTexts, testTexts, trainLabels, testLabels = train_test_split(dataset['text'], dataset['label'], test_size=0.2)


vectorizer = CountVectorizer(stop_words='english', max_features=10000)
XTrainBow = vectorizer.fit_transform(trainTexts)
XTestBow = vectorizer.transform(testTexts)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

modelBow = LogisticRegression(max_iter=1000)
modelBow.fit(XTrainBow, trainLabels)

bowPredictions = modelBow.predict(XTestBow)
accuracy_bow = accuracy_score(testLabels, bowPredictions)
f1_bow = f1_score(testLabels, bowPredictions)

print(f"BoW Model Accuracy: {accuracy_bow}, F1-Score: {f1_bow}")




import numpy as np
# EMBEDDINGS DATASET
def document_to_embedding_avg(document, model):
    embeddings = [model.wv[word] for word in word_tokenize(document.lower()) if word in model.wv and word not in stopwords.words('english')]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

XTrainEmbeddings = np.array([document_to_embedding_avg(doc, sgModel) for doc in trainTexts])
XTestEmbeddings = np.array([document_to_embedding_avg(doc, sgModel) for doc in testTexts])


model_embeddings = LogisticRegression(max_iter=1000)
model_embeddings.fit(XTrainEmbeddings, trainLabels)

embeddingsPredictions = model_embeddings.predict(XTestEmbeddings)
accuracy_embeddings = accuracy_score(testLabels, embeddingsPredictions)
f1_embeddings = f1_score(testLabels, embeddingsPredictions)

print(f"Embeddings Model Accuracy: {accuracy_embeddings}, F1-Score: {f1_embeddings}")