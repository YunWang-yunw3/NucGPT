# Import the libraries
import torch
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
import os
import faiss


def query_message(
    query: str,
    data: list,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    # introduction = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    introduction = 'Use the below relevant information on the nuclear fission reactor to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for datum in data:
        next_article = f'\nRelevant Information:{datum}\n'
        # if (
        #     num_tokens(message + next_article + question, model=model)
        #     > token_budget
        # ):
        #     break
        # else:
        message += next_article
    return message + question

def ask(
    query: str,
    data: list,
    model: str = 'RWKV-14B',
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and with relevant information."""
    message = query_message(query, data, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    prompt = f'You are answering questions about nuclear fission reactor.\n{message}\nHere is your answer:\n'
    return prompt
    # messages = [
    #     {"role": "system", "content": "You answer questions about nuclear fission reactor."},
    #     {"role": "user", "content": message},
    # ]
    # response = run_model(
    #     model=model,
    #     messages=messages,
    #     temperature=0.7,
    # )
    # response_message = response["choices"][0]["message"]["content"]
    # return response_message
    # return response


if __name__ == '__main__':
    os.environ['RWKV_JIT_ON'] = '1'
    os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
    commandline_args = os.environ.get('COMMANDLINE_ARGS', "--skip-torch-cuda-test --no-half")

    # Load the index and text from the files
    print("Loading Index and Text...")
    index = faiss.read_index('./index_data/book-001.index')
    texts = open('./index_data/book-001_text.txt', "r").read().splitlines()
    filenames = open('./index_data/book-001_filenames.txt', "r").read().splitlines()
    page_numbers = open('./index_data/book-001_page_numbers.txt', "r").read().splitlines()
    # print(texts[1680])

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    def get_embedding(text):
        tokens = tokenizer(text, return_tensors="pt")
        output = bert_model(**tokens)
        # Get the mean of the last hidden states across tokens
        embedding = torch.mean(output.last_hidden_state, dim=1)
        # Return the embedding as a numpy array
        return embedding.detach().numpy()
    
    print("\nLoading LLM models...")
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE, PIPELINE_ARGS
    llm_model_path = input("Enter the path to the LLM model: ")
    # llm_model = RWKV(model='models/RWKV-4b-Pile-171M-20230202-7922.pth', strategy='cpu bf16')
    llm_model = RWKV(model=llm_model_path, strategy='cpu bf16')
    pipeline = PIPELINE(llm_model, "20B_tokenizer.json") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV

    # Define a loop to wait for human typed input and print the results
    while True:
        # Get the input from the user
        usr_input = input("Enter your query or type 'stop' to quit: ")

        # Check if the user wants to quit or stop
        if usr_input.lower() in ['quit', 'stop']:
            break

        # Convert the input to an embedding vector
        input_emb = get_embedding(usr_input)

        # Search the index with the query vector
        k = 5 # Number of nearest neighbors
        distances, ids = index.search(input_emb.reshape(1, -1), k) # distances contains distances, ids contains ids
        print('Relevant information at: ', ids, '\n')
        # Use the ids to retrieve the linked texts
        results = []
        relevant_texts = []
        for i in range(k):
            idx = ids[0][i]
            text = texts[idx]
            if idx > 0 & idx < len(texts)-1:
                relevant_texts.append(texts[idx-1])
            relevant_texts.append(text)
            if idx < len(texts)-1:
                relevant_texts.append(texts[idx+1])
            distance = distances[0][i]
            filename = filenames[idx]
            page_number = page_numbers[idx]
            results.append((text, distance, filename, page_number))

        # Ask the question
        prompt = ask(usr_input, relevant_texts)
        print(prompt)
        def my_print(s):
            print(s, end='', flush=True)
        args = PIPELINE_ARGS(temperature = 1.0, top_p = 1.0, top_k = 100, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)
        pipeline.generate(prompt, token_count=200, args=args, callback=my_print)
        print('\n')

        out, state = llm_model.forward([187, 510, 1563, 310, 247], None)
        # print(out.detach().cpu().numpy())                   # get logits
        out, state = llm_model.forward([187, 510], None)
        out, state = llm_model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
        out, state = llm_model.forward([310, 247], state)
        # print(out.detach().cpu().numpy())                   # same result as above
        print('\n')


        # Print the results in a nice format
        # print(f"Top {k} results for '{usr_input}':")
        # for i, (text, distance, filename, page_number) in enumerate(results):
        #     print(f"{i+1}. Text: {text}")
        #     print(f"   Distance: {distance}")
        #     print(f"   Filename: {filename}")
        #     print(f"   Page number: {page_number}")
        # print(f"\nAnswer: {answer}\n\n")
        # rel_text = '\n\n\n'.join(relevant_texts)
        # print(rel_text)
