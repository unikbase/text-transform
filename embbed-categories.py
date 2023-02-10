import pickle
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

#Our sentences we like to encode
sentences = ['Art',
    'Sculpture', 
    'Numismatic','Ingots','Furs','Kitchenware','Plates']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
#for sentence, embedding in zip(sentences, embeddings):
#    print("Sentence:", sentence)
#    print("Embedding:", embedding)
#    print("")

with open('categories.pickle', 'wb') as f:
    pickle.dump(zip(sentences,embeddings), f)
