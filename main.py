# Yeah
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from datasets import load_dataset
# dataset = load_dataset("wikipedia", "20220301.simple")


# # THIS BIT WAS FOR FIRST RUN TO DOWNLOAD WORD2VEC FILES

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


# After models have been saved with 1000 wikipedia examples



from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# # Load the CBOW and Skip-gram models
# cbowModel = Word2Vec.load("model_cbow.word2vec")
# sgModel = Word2Vec.load("model_sg.word2vec")

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




# Bias in word embeddings - WEAT from WEFE
from wefe.datasets import load_bingliu
from wefe.metrics import RNSB
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel

import pandas as pd
import plotly.express as px
import gensim.downloader as api

# load the target word sets.
# In this case each word is an objective set because each of them represents a different social group.
ethnicity_nationality_words = [
    ["african", "asian", "hispanic", "native"],  # Groups typically considered minorities
    ["european", "american", "western"],  # Groups often associated with the majority or dominant culture
]

religion_related_words = [
    ["christian", "islam", "judaism", "buddhism", "hinduism"],
    ["atheist", "agnostic", "secular"],
]

bing_liu = load_bingliu()

# Create the query
query = Query(religion_related_words, [bing_liu["positive_words"], bing_liu["negative_words"]])



def evaluate(
    query: Query, gensim_model_name: str, short_model_name: str, model_args: dict = {}
):
    # Fetch the model
    model = WordEmbeddingModel(
        api.load(gensim_model_name), short_model_name, **model_args
    )

    # Run the queries
    results = RNSB().run_query(
        query, model, holdout=True, print_model_evaluation=True, n_iterations=100
    )

    # Show the results obtained with glove
    fig = px.bar(
        pd.DataFrame(
            results["negative_sentiment_distribution"].items(),
            columns=["Word", "Sentiment distribution"],
        ),
        x="Word",
        y="Sentiment distribution",
        title=f"{short_model_name} Negative Sentiment Distribution",
    )

    fig.update_yaxes(range=[0, 0.2])
    fig.show()


evaluate(
    query,
    "conceptnet-numberbatch-17-06-300",
    "Conceptnet",
    model_args={"vocab_prefix": "/c/en/"},
)



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

# # Flatten the list of words to a single list
# all_religion_words = sum(religion_related_words, [])
# cbow_not_found = check_coverage(cbowModel, all_religion_words)
# sg_not_found = check_coverage(sgModel, all_religion_words)

# print("CBOW Model missing words:", cbow_not_found)
# print("Skip-gram Model missing words:", sg_not_found)

# # Assuming cbowModel and sgModel are already loaded Gensim Word2Vec models
# evaluate_with_preloaded_model(query, cbowModel, "CBOW Model")
# evaluate_with_preloaded_model(query, sgModel, "Skip-gram Model")



