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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
import argparse
from tqdm import tqdm



class RagFromScratch():
    def __init__(self, embedding_model, rerank_model, file_path):
        super().__init__()
        self.file_path = file_path
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.embedding_model = SentenceTransformer(model_name_or_path = embedding_model, device=self.device)
        self.rerank_model = CrossEncoder(rerank_model)
        
        # self._run()


    def run(self):
        # get the document: using a pdf file as a placeholder for private documents, but it could be anything - emails, docs, etc
        if not os.path.exists(self.file_path):
            print('File does not exist.')
            return
        else:
            print(self.file_path)
            self.doc = fitz.open(self.file_path)


        # store the document into a dictionary with relevant metadata
        self.pages_and_texts = []
        for page_number, page in tqdm(enumerate(self.doc)):
            text = page.get_text()
            text = text.replace('\n', ' ').strip()
            self.pages_and_texts.append({
                'page_number' : page_number - 41,
                'page_char_count' : len(text),
                'page_word_count' : len(text.split(' ')) if len(text) > 0 else 0,
                'page_sentence_count' : len(text.split('. ')) if len(text) > 0 else 0,
                'page_token_count' : len(text)/4, 
                'text' : text
            })
        
        # print(f'pagse and texts : {self.pages_and_texts}')

        # Split text into sentences using a sentencizer from Spacy
        self._text2sentence()
        # print(self.pages_and_texts)

        # Further group the sentences into chunks to adapt to the context window.
        self.pages_and_chunks = self._create_chunks(10)

        # Filter out the chunks that do not have any useful information. One way to do that is to remove chunks less than min_length. We can also use overlap here, so that all the beginning of one chunk and the end is always a part of the previous and the next chunk. In this way, most of the text is preserved from the original document.
        self.pages_and_chunks_over_min_token_len = self._filter_chunks()

        # get documents to embeddings-numerical representations of the text
        self.text_chunk_embeddings = self._create_embeddings()

        # add the embeddings to a vector store - we use vector store from the FAISS library.
        self._create_vector_store()
        self._load_llm()
        return


    def ask(self, query,
            temperature=0.7,
            max_new_tokens=256):
        '''
        Takes the query, finds the relevant resources and generates the answer to the query based on the relevant resources from the private documents.
        '''

        # get the scores and indices from RAG
        self.context_list = self._retrieve_relevant_resources(query, 5)
        self.prompt = self._prompt_formatter(query)

        # Create a chat template
        chat = [{
            'role' : 'user',
            'content' : self.prompt
            }]
        self.prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # tokenize the input text
        input_ids = self.tokenizer(self.prompt, return_tensors='pt').to(self.device)

        # Generate output from the local LLM
        outputs = self.llm_model.generate(**input_ids, 
                                temperature=temperature,
                                do_sample=True,
                                max_new_tokens=max_new_tokens)
        
        outputs_decoded = self.tokenizer.decode(outputs[0])
        outputs_decoded = outputs_decoded.replace(self.prompt, '').replace('<bos>', '').replace('<eos>', '')

        # if you want to check the sources of the generated text for debugging:
        return outputs_decoded, self.context_list

        # return outputs_decoded
    

    def _text2sentence(self):
        nlp = English()

        # Add a sentencizer pipeline
        nlp.add_pipe('sentencizer')
        for item in self.pages_and_texts:
            
            item['sentences'] = list(nlp(item['text']).sents)
            
            # make sure all sentences are string. Default is a spacy datatype.
            item['sentences'] = [str(strng) for strng in item['sentences']]
            item['sentences_per_page'] = len(item['sentences'])

        return

    

    def _create_chunks(self, slice_size):
        # Define split size to turn groups of sentences into chunks
        num_sentence_chunk_size = slice_size
        
        for item in self.pages_and_texts:
            item['sentence_chunks'] = [item['sentences'][i:i+num_sentence_chunk_size] for i in range(0, len(item['sentences']), num_sentence_chunk_size)]
            item['num_chunks'] = len(item['sentence_chunks'])


        # creating chunk dict
        pages_and_chunks = []
        for item in self.pages_and_texts:
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

        return pages_and_chunks

    def _filter_chunks(self):
        # Filter chunks with very small text length. These chunks might not have useful information.
        min_token_length = 30
        df = pd.DataFrame(self.pages_and_chunks)
        pages_and_chunks_over_min_token_len = df[df['chunk_token_count'] > min_token_length].to_dict(orient='records')

        return pages_and_chunks_over_min_token_len

    def _create_embeddings(self):
        text_chunks = [item['sentence_chunk'] for item in self.pages_and_chunks_over_min_token_len]
        text_chunk_embeddings = self.embedding_model.encode(
            text_chunks,
            batch_size = 32,
            convert_to_tensor=True
            )
        
        return text_chunk_embeddings
    
    def _create_vector_store(self):
        # Convert the tensor from GPU to CPU and detach it from the graph
        # Then convert to a numpy array of type float32
        self.text_chunk_embeddings = np.array(self.text_chunk_embeddings.cpu(), dtype=np.float32)
        d = 768

        index_path = "./my_faiss.index"
        if not os.path.exists(index_path):
            # setting up the vector store:
            self.index = faiss.IndexFlatL2(d)
            self.index.add(self.text_chunk_embeddings)
            faiss.write_index(self.index, index_path)  # Save the index
        else:
            # Directly load from disk
            self.index = faiss.read_index(index_path)



    
    def _retrieve_relevant_resources(self, query, num_res_to_return):
        '''
        Embeds a query with the used model and returns top k scores and indices from vector store.
        '''

        # embed the query
        query_embed = self.embedding_model.encode(query, convert_to_tensor=True)
        query_embed = query_embed.cpu().reshape(1,-1)
        D,I = self.index.search(query_embed, num_res_to_return+5)
        retreived_docs = [self.pages_and_chunks_over_min_token_len[idx]['sentence_chunk'] for idx in I[0]]
        
        return self.rerank_model.rank(query, retreived_docs, return_documents=True, top_k=num_res_to_return)


    def _prompt_formatter(self, query):
        '''
        Prompting techniques to use : 
        1.Give clear intructions.
        2.Give a few input/output examples:(Manual COT)
        3.Ask to work the query, step by step:(Automatic COT), Give step by step reasoning.
        '''

        context = '-'+'\n-'.join([item['text'] for item in self.context_list])
        base_prompt = """
        Based on the following context items, please answer the query.
        Give yourself room to think by extracting relevant passages from the context before answering the query.
        Don't return the thinking, only return the answer.
        Make sure your answers are as explanatory as possible.
        Use the following examples as reference for the ideal answer style.
        \nExample 1:
        Query: What are the fat-soluble vitamins?
        Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
        \nExample 2:
        Query: What are the causes of type 2 diabetes?
        Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
        \nExample 3:
        Query: What is the importance of hydration for physical performance?
        Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
        \nNow use the following context items to answer the user query:
        {context}
        \nRelevant passages: <extract relevant passages from the context here>
        User query: {query}
        Answer:
        """
        prompt = base_prompt.format(context = context, query=query)
        return prompt


    def _load_llm(self):
        # 1. Create quantization config for smaller model loading (optional)
        # Requires !pip install bitsandbytes accelerate, see: https://github.com/TimDettmers/bitsandbytes, https://huggingface.co/docs/accelerate/
        # For models that require 4-bit quantization (use this if you have low GPU memory available)

        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                bnb_4bit_compute_dtype=torch.float16)


        # Flash Attention 2 requires NVIDIA GPU compute capability of 8.0 or above, see: https://developer.nvidia.com/cuda-gpus
        # Requires !pip install flash-attn, see: https://github.com/Dao-AILab/flash-attention 
        if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"
        print(f"[INFO] Using attention implementation: {attn_implementation}")

        # 2. Pick a model we'd like to use (this will depend on how much GPU memory you have available)
        model_id = "google/gemma-7b-it"
        model_id = model_id 
        print(f"[INFO] Using model_id: {model_id}")

        # 3. Instantiate tokenizer (tokenizer turns text into numbers ready for the model) 
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

        # 4. Instantiate the model
        self.llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                        torch_dtype=torch.float16, # datatype to use, we want float16
                                                        #  quantization_config=quantization_config if use_quantization_config else None,
                                                        low_cpu_mem_usage=False, # use full memory 
                                                        attn_implementation=attn_implementation) # which attention version to use

        # if not use_quantization_config: # quantization takes care of device setting automatically, so if it's not used, send model to GPU 
        self.llm_model.to(self.device)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', action='store', dest='prompt', default=None)
    args = parser.parse_args()

    pdf_path = './Rag-From-Scratch/simple-local-rag/human-nutrition-text.pdf'
    embedding_model = 'all-mpnet-base-v2'
    reranking_model = 'mixedbread-ai/mxbai-rerank-large-v1'


    rag_from_scratch = RagFromScratch(embedding_model, reranking_model, pdf_path)
    rag_from_scratch.run()

    # augment the query and generate from the LLM
    response = rag_from_scratch.ask(args.prompt)
    print(response)



