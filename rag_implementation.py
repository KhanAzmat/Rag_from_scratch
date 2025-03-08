import fitz
import os
from spacy.lang.en import English
import re
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sentence_transformers import CrossEncoder

class ragFromScratch():
    def __init__(self, embedding_model, rerank_model, docs):




def text2sentence(pages_and_text):
    nlp = English()

    # Add a sentencizer pipeline
    nlp.add_pipe('sentencizer')
    for item in pages_and_text:
        item['sentences'] = list(nlp(item['text']).sents)
        
        # make sure all sentences are string. Default is a spacy datatype.
        item['sentences'] = [str(strng) for strng in item['sentences']]
        item['sentences_per_page'] = len(item['sentences'])

    return pages_and_texts

def create_chunks(pages_and_text):
    # Define split size to turn groups of sentences into chunks
    num_sentence_chunk_size = 10
    
    for item in pages_and_text:
        item['sentence_chunks'] = [item['sentences'][i:i+num_sentence_chunk_size] for i in range(0, len(item['sentences']), num_sentence_chunk_size)]
        item['num_chunks'] = len(item['sentence_chunks'])


    # creating chunk dict
    pages_and_chunks = []
    for item in pages_and_text:
        for sentence_chunk in item['sentence_chunks']:  
            chunk_dict = {}
            chunk_dict['page_number'] = item['page_number']
            joined_sentence_chunk = ''.join(sentence_chunk).replace('  ', ' ').strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)

            chunk_dict['sentence_chunk'] = joined_sentence_chunk
            chunk_dict['chunk_char_count'] = len(joined_sentence_chunk)
            chunk_dict['chunk_word_count'] = len(joined_sentence_chunk.split(' '))
            chunk_dict['chunk_token_count'] = len(joined_sentence_chunk)/4

            pages_and_chunks.append(chunk_dict)

    # Filter chunks with very small text length. These chunks might not have useful information.
    min_token_length = 30
    df = pd.DataFrame(pages_and_chunks)
    pages_and_chunks_over_min_token_len = df[df['chunk_token_count'] > min_token_length].to_dict(orient='records')


def create_embeddings(embedding_model):
    text_chunks = [item['sentence_chunk'] for item in pages_and_chunks_over_min_token_len]
    text_chunk_embeddings = embedding_model.encode(
        text_chunks,
        batch_size = 32,
        convert_to_tensor=True
        )

def create_vector_store():
    # Convert the tensor from GPU to CPU and detach it from the graph
    # Then convert to a numpy array of type float32
    text_chunk_embeddings = np.array(text_chunk_embeddings.cpu(), dtype=np.float32)
    d = 768

    # setting up the vector store:
    index = faiss.IndexFlatL2(d)
    index.add(text_chunk_embeddings)

def top_k_results():
    D,I = index.search(query_embed, k)
    retreived_docs = [pages_and_chunks_over_min_token_len[idx]['sentence_chunk'] for idx in I[0]]
    results = reranking_model.rank(query, retreived_docs, return_documents=True, top_k=3)

if __name__=='main':
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    pdf_path = './Rag-From-Scratch/simple-local-rag/human-nutrition-text.pdf'
    embedding_model = SentenceTransformer(model_name_or_path = 'all-mpnet-base-v2', device=device)
    reranking_model = CrossEncoder('mixedbread-ai/mxbai-rerank-large-v1')

    query = 'macronutrients functions'

    # embed the query
    query_embed = embedding_model.encode(query, convert_to_tensor=True)
    query_embed = query_embed.cpu().reshape(1,-1)

    if not os.path.exists(pdf_path):
        print('File does not exist.')
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text.replace('\n', ' ').strip()
        pages_and_texts.append({
            'page_number' : page_number - 41,
            'page_char_count' : len(text),
            'page_word_count' : len(text.split(' ')) if len(text) > 0 else 0,
            'page_sentence_count' : len(text.split('. ')) if len(text) > 0 else 0,
            'page_token_count' : len(text)/4, 
            'text' : text
        })
    pages_and_texts = map(pages_and_texts, text2sentence)