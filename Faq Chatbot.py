pip install sentence-transformers pandas scikit-learn

# Commented out IPython magic to ensure Python compatibility.
# %%writefile faqs.csv
# question,answer
# What is your name?,I am an FAQ chatbot.
# What are your working hours?,We work from 9 AM to 6 PM.
# How can I contact support?,Email support@example.com.
# What services do you provide?,We provide AI and software solutions.

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the FAQs from the CSV file
df = pd.read_csv('faqs.csv')

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the answers
answer_embeddings = model.encode(df['answer'].tolist())

print("FAQ data loaded and embeddings generated.")
print(f"Number of FAQs: {len(df)}")
print(f"Shape of answer embeddings: {answer_embeddings.shape}")

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load FAQ data
faq = pd.read_csv("faqs.csv") # Corrected filename from 'faq.csv' to 'faqs.csv'

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings for questions
question_embeddings = model.encode(
    faq["question"].tolist(),
    convert_to_numpy=True
)

print("FAQ Chatbot Ready!")

while True:
    user_query = input("\nYou: ")

    if user_query.lower() in ["exit", "quit"]:
        break

    query_embedding = model.encode(
        [user_query],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_embedding,
        question_embeddings
    )[0]

    best_match = np.argmax(similarities)

    if similarities[best_match] > 0.4:
        print("Bot:", faq.iloc[best_match]["answer"])
    else:
        print("Bot: Sorry, I couldn't find an answer.")
