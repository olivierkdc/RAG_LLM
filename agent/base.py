import os 
import json
import re 
import unicodedata
import subprocess
import numpy as np 
from sentence_transformers import SentenceTransformer
import faiss
from .config import BASE_DIR, LOG_DIR, SCHEMA_DIR, PROMPT_DIR, AI_AGENT_MODEL, EMBEDDER_MODEL

class AI_Agent():
    def __init__(self):
        self.role = "Generic AI Agent"
        self.base_dir = BASE_DIR
        self.log_dir = LOG_DIR
        self.schema_dir = SCHEMA_DIR
        self.prompt_dir = PROMPT_DIR

    def retrieve_chunks(self, query, index, metadata, model):
        # Encode the query
        query_embedding = model.encode([query], convert_to_tensor=False)[0]

        # Search FAISS for the top 3 most similar chunks
        top_k = 3
        distances, indices = index.search(np.array([query_embedding], dtype="float32"), top_k)

        # Retrieve the corresponding chunks from metadata
        results = []
        for idx in indices[0]:
            if idx < len(metadata):
                results.append(metadata[idx]["chunk"])

        return results


    def create_prompt(self, user_query, retrieved_chunks = None):
        # Combine retrieved chunks into a clear and structured context
        if retrieved_chunks:
            context = "\n\n".join(retrieved_chunks)
        else: 
            context = ""

        # Create a more explicit and task-specific prompt
        prompt_file_path = str(self.prompt_dir) + '/financial_prompt.txt'
        with open(prompt_file_path) as f:
            prompt = f.read()
            prompt = prompt.format(
                user_query = user_query,
                context = context,
            )
        return prompt
    

    def generate_answer(self, prompt, model_name=AI_AGENT_MODEL):
        try:
            # Run the subprocess command
            result = subprocess.run(
                ["ollama", "run", model_name],
                input=prompt,  # Provide the prompt as input
                text=True,  # Treat input/output as text
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE,  # Capture errors
                encoding="utf-8"
            )
            # Return the standard output
            return result.stdout.strip()
        except Exception as e:
            print("Error during Ollama subprocess call:", str(e))
            return ""


    def run(self, user_query, verbose):
        model = SentenceTransformer(EMBEDDER_MODEL)
        index = faiss.read_index("data/embedded/financial_report_index.faiss")
        with open("data/embedded/financial_report_metadata.json", "r", encoding="utf-8") as file:
            metadata = json.load(file)
        retrieved_chunks = self.retrieve_chunks(query = user_query, index = index, metadata = metadata, model = model)
        final_prompt = self.create_prompt(user_query = user_query, retrieved_chunks=retrieved_chunks)
        if verbose: print(final_prompt)
        answer = self.generate_answer(prompt = final_prompt)
        print(answer)
        return 