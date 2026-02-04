import sys 
from time import time 
import os
import numpy as np
import re
import json
from bs4 import BeautifulSoup
import html
import nltk
# nltk.download('punkt_tab') # Run as necessary
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import subprocess

from agent.config import EMBEDDER_MODEL
from agent.base import AI_Agent

def prepare_RAG_dataset(company_name = "freddie_mac"):
    """
    Runs necessary steps to prepare the vector database for the RAG process.
    If this functione is not executed first, we will be querying an empty database and the agent will not have the context to support answers.
    """
    print("Preparing RAG Dataset")
    
    # Load the text file
    data_directory = "data"
    file_name = f"{company_name}.txt"
    file_path = data_directory + "/raw/" + file_name
    
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()
    print(raw_text[:500])

    def clean_text(text):
        # Remove HTML tags
        for parser in ["lxml", "html5lib", "html.parser"]:
            try: 
                text = BeautifulSoup(text, parser).get_text()
                # Decode HTML-encoded characters
                text = html.unescape(text)
                # Remove extra whitespace
                text = re.sub(r"\s+", " ", text.strip())
                return text
            except:
                continue
        return text 
    cleaned_text = clean_text(raw_text)
    print("Sample cleaned text:", cleaned_text[:500])  # Verify the cleaning process

    # Tokenize into sentences (you may need to install nltk: pip install nltk)
    sentences = sent_tokenize(cleaned_text)

    # Group sentences into chunks of ~500 words
    chunk_size = 500
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(" ".join(current_chunk)) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:  # Add remaining sentences
        chunks.append(" ".join(current_chunk))

    print(f"Total chunks: {len(chunks)}")
    print(chunks[0])  # Check the first chunk

    # Save chunks as a JSON file
    output_file = data_directory + "/embedded/" + "financial_report_chunks.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=4)

    print(f"Chunks saved to {output_file}")

    # Load a pre-trained model
    model = SentenceTransformer(EMBEDDER_MODEL)  # Ensure you utilize the same embedder model as the Agent down the line.
    
    # Generate embeddings for all chunks
    embeddings = model.encode(chunks, convert_to_tensor=False)

    print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")

    # Convert embeddings to a NumPy array
    embedding_dim = len(embeddings[0])  # Get embedding dimensions
    embedding_array = np.array(embeddings, dtype="float32")

    # Initialize a FAISS index
    # index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity search
    index = faiss.IndexFlatIP(embedding_dim)  # cosine similarity for similarity search

    # Add embeddings to the index
    index.add(embedding_array)

    # Save the index for future use
    faiss.write_index(index, data_directory + "/embedded/" + "financial_report_index.faiss")

    print("Embeddings stored in FAISS index.")

    metadata = [{"chunk": chunk, "id": idx} for idx, chunk in enumerate(chunks)]

    # Save metadata as JSON
    metadata_file = data_directory + "/embedded/" + "financial_report_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=4)

    print(f"Metadata saved to {metadata_file}")

def execute_LLM_agent(verbose = False):
    """
    execute_LLM_agent.
    Generates a user experience to pass in the user_query from the terminal.
    :param verbose: Description turns on verbosity for the run of the agent. This will print the final prompts passed to the Agent.
    """
    print("Executing LLM")
    AI = AI_Agent(prompt_type = 'financial')
    print("Please utilize the AI Agent! Type 'exit' to quit.")
    while True:
        user_query = input('\nEnter your natural language query:\n')

        if user_query.lower() in ["exit","quit","stop","break"]:
            print("Session Ended.")
            break
        AI.run(user_query, verbose = verbose)

def benchmark_agent(verbose = True):
    """
    """
    print("Evaluating LLM against testing dataset")
    Evaluation_AI = AI_Agent(prompt_type='evaluation')
    Generator_AI = AI_Agent(prompt_type = 'financial')
    test_question_file_path = "./agent/evaluation/evaluation.json"
    with open(test_question_file_path) as json_file:
        evaluation_data = json.load(json_file)    
    
    for question in evaluation_data:
        data = evaluation_data[question]
        user_query = data['question']
        ground_truth = data['answer']

        if verbose:
            print(data)

        agent_answer = Generator_AI.run(user_query, verbose = False)
        evaluation_answer = Evaluation_AI.run(user_query,ground_truth=ground_truth,agent_answer=agent_answer, verbose = False)
    return 


def main():
    if len(sys.argv) == 1:
        print("Please provide a job_type parameter. Pass 'LLM' to chat with the agent, or pass 'RAG' to prepare the dataset.") 
    else:
        job_type = sys.argv[1].lower()
        if job_type in ('agent','llm','chat'):
            execute_LLM_agent()
        elif job_type in ('rag','data','preparation','prep'):
            prepare_RAG_dataset()
        elif job_type in ('evaluation','evaluate','validation','validate','test','benchmark'):
            benchmark_agent()
        else:
            print("Please specify a proper job_type. Pass 'LLM' to chat with the agent, or pass 'RAG' to prepare the dataset.")

if __name__ == '__main__':
    main()