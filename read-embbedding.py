import pickle

with open('categories.pickle', 'rb') as f:
     categories_embeddings = pickle.load(f)

for sentence, embedding in categories_embeddings:
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

