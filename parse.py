# Import the necessary modules
import torch
import math
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import os
import faiss
import numpy as np


# Define a function that takes a folder path and returns a list of embeddings for each file
def get_embeddings_and_texts_from_folder(folder_path):
    # Define a function to get the embedding of a text

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    def get_tokenized_text(text):
        # Tokenize the text and convert to tensor
        tokens = tokenizer(text, return_tensors="pt")
        # Return the tokens
        # token_size = len(tokens["input_ids"])
        return tokens
    def get_embedding(tokens):
        # Get the output of the last hidden layer of BERT
        output = model(**tokens)
        # Get the mean of the last hidden states across tokens
        embedding = torch.mean(output.last_hidden_state, dim=1)
        # Return the embedding as a numpy array
        return embedding.detach().numpy()
    
    def split_string(string, chunk_size, overlap):
        chunks = []
        for i in range(0, len(string), chunk_size - overlap):
            chunk = string[i:i + chunk_size]
            chunk = chunk.strip().replace('\n', ' ')
            chunks.append(chunk)
        return chunks


    # Create a TextSplitter object with the desired chunk size and overlap
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 600,
        chunk_overlap  = 120,
        length_function = len,
    )

    # Create an empty list to store the embeddings and the texts
    embeddings = []
    texts = []
    filenames = []
    page_numbers = []

    # Loop over the files in the folder
    for filename in os.listdir(folder_path):
        if filename in ['book-001-0---Contents_2021_High-Temperature-Gas-Cooled-Reactors.pdf', 'book-001-0---Title-page_2021_High-Temperature-Gas-Cooled-Reactors.pdf', 'book-001-6---Index_2021_High-Temperature-Gas-Cooled-Reactors.pdf']:
            continue

        # Check if the file is a PDF file
        if filename.endswith(".pdf"):
            print(f"\nProcessing {filename}...")
            # Read the PDF file and extract the text
            pdf_reader = PdfReader(os.path.join(folder_path, filename))
            for i, page in enumerate(pdf_reader.pages):
                # text = pdf_reader.getPage(0).extract_text()
                text = page.extract_text()

                # chunk_size = 600
                # try:
                #     chunk_size = math.ceil(len(text) / (len(text) // 600)) # 600 for now, 1000 maybe in the future with edge case
                # except:
                #     pass
                # overlap = math.ceil(chunk_size * 0.2)

                # # Split the text into chunks using the TextSplitter (stupid), we use split_string
                # chunks = split_string(text, chunk_size, overlap)
                chunks = text_splitter.split_text(text)

                # Loop over the chunks and perform text embedding
                for chunk in chunks:
                    # check token length, TODO: in the future
                    chunk = chunk.strip().replace('\n', ' ')
                    tokens = get_tokenized_text(chunk)
                    # num_tokens = len(tokens["input_ids"])
                    # assert(num_tokens <= 512)

                    # Call the get_embedding function with the chunk and the number of tokens
                    embedding = get_embedding(tokens)

                    # Append the embedding to the list of embeddings
                    embeddings.append(embedding)
                    texts.append(chunk)
                    filenames.append(filename)
                    page_numbers.append(i)
            # return embeddings, texts, filenames, page_numbers

    # Return the list of embeddings
    return embeddings, texts, filenames, page_numbers

# Define a function that takes a folder path and creates a FAISS index with embeddings and texts
def create_faiss_index_from_folder(folder_path):
    # Define the dimension and metric of your embeddings
    dim = 768 # For example, using BERT embeddings
    # metric = faiss.METRIC_INNER_PRODUCT # For example, using cosine similarity

    # Create an index object in FAISS
    index = faiss.IndexFlatIP(dim)

    embeddings, texts, filenames, page_numbers = get_embeddings_and_texts_from_folder(folder_path)
    print(texts[200])

    # Convert the list of embeddings to a numpy array
    embeddings = np.array(embeddings).astype("float32").reshape(-1, dim)
    print(embeddings.shape)

    # Add the embeddings to the index
    index.add(embeddings)

    # Optionally, save the index and the texts to files
    faiss.write_index(index, "./index_data/book-001.index")
    with open("./index_data/book-001_text.txt", "w") as f:
        for text in texts:
            f.write(text + "\n")
    # f.close()
    with open("./index_data/book-001_filenames.txt", "w") as f:
        for filename in filenames:
            f.write(filename + "\n")
    # f.close()
    with open("./index_data/book-001_page_numbers.txt", "w") as f:
        for page_number in page_numbers:
            f.write(str(page_number) + "\n")
    # f.close()

    # Return the index and the texts
    return index, texts


if __name__ == '__main__':
    import time
    start_time = time.time()

    create_faiss_index_from_folder('./HGTR_books/book-001')

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"The function took {elapsed_time} seconds to run")
