# RAG-Powered Chatbot: Financial Report Q&A Assistant

This project is a **Retrieval-Augmented Generation (RAG)** powered chatbot designed to answer questions based on unstructured data. Using **Financial Reports obtained from SEC** as an example dataset, the chatbot demonstrates how modern AI techniques can extract relevant insights from complex documents.

---

## 🌟 Features
- **Contextual Q&A:** Provides accurate answers based on retrieved context.
- **RAG Architecture:** Combines vector-based retrieval with a language model for generation.
- **Open-Source Tools:** Built with FAISS, Sentence Transformers, and Ollama.
- **Customizable:** Easily adaptable to other datasets or domains.

---

## 🚀 How It Works
1. **Data Preparation:**
   - The financial report is cleaned, chunked into manageable pieces, and indexed.
2. **Embedding Generation:**
   - Dense embeddings are created using the `all-MiniLM-L6-v2` model from Sentence Transformers.
3. **Context Retrieval:**
   - FAISS retrieves the most relevant chunks for a user query.
4. **Answer Generation:**
   - Ollama's LLM generates a concise and factual answer based on the retrieved context.
---

## 📦 Installation

### Prerequisites
- Python 3.8 or later
- [Ollama](https://ollama.ai/) installed locally

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/olivierkdc/RAG_LLM.git
   cd RAG_LLM
   ```   
2. **Install Dependencies**: 
 - Install the required Python libraries:
    ```bash
    pip install sentence-transformers faiss-cpu beautifulsoup4 nltk numpy
    ```
3. **Prepare the Dataset**:
 - Place your unstructured text data (e.g., freddie_mac.txt) in the data/raw/ directory.
  
4. **Run the Pipeline**: 
 - Execute the RAG pipeline to query the chatbot, specifying the job_type:
   - pass RAG to process the data/raw. current system hardcodes the relevant file in the raw data.
   - pass LLM to utilize the chatbot. 
    ```bash
    python app.py RAG
    ```          
    ```bash
    python app.py LLM
    ```          
---

## 📚 Requirements

1. Below are the dependencies required for the project:
   Depencies can be found inside the requirements.txt file.
   
 - **Sentence Transformers**: sentence-transformers
 - **FAISS**: faiss-cpu
 - **BeautifulSoup (for cleaning HTML)**: beautifulsoup4
 - **NLTK**: nltk
 - **NumPy**: numpy

2. Install all dependencies with:
    ```bash
    pip install -r requirements.txt
    ```
---

## 🙌 Acknowledgments
 - Sentence Transformers for embedding generation.
 - Ollama for local LLM-based answer generation.
 - FAISS for efficient similarity search.
---

## 🌟 Future Enhancements
 - Add support for multi-turn conversations.
 - Integrate other open-source LLMs for generation.
 - Enable deployment to cloud platforms for scalability.

### Quick Links
Obtain data from the following.
Current setup relies on copying financial reports and saving as text files to be processed.
- https://www.sec.gov/search-filings 

---
## **Project Overview**
Build a Q&A chatbot that utilizes Retrieval-Augmented Generation (RAG) to provide accurate and contextually relevant answers to customer queries. The chatbot dynamically retrieves domain-specific knowledge and integrates it with a Large Language Model (LLM) to generate responses.

### **Key Features**
- **RAG Integration:** Combines vector-based knowledge retrieval with LLMs for dynamic and accurate query responses.
- **Prompt Engineering:** Optimized prompts to improve response quality by effectively utilizing retrieved knowledge.
- **Interactive Demo:** A user-friendly interface to showcase real-time chatbot functionality.

### **Technologies Used**
- **Vector Database:** Pinecone, Weaviate, or FAISS for document retrieval.
- **LLM Backend:** Open-source LLM (e.g., LLaMA or similar) integrated with prompt engineering.
- **Embedding Models:** OpenAI embeddings or Sentence Transformers for document and query vectorization.
- **Frontend:** Streamlit for a web-based interface or CLI for lightweight interaction.
- **Programming Language:** Python.

### **Outcome**
- A GitHub-hosted project with:
  - Complete source code.
  - Interactive demo (web or CLI-based). (Work in Progress)
  - Documentation explaining implementation details and usage.

### **Applications**
- Customer support automation for e-commerce, insurance, and other industries.
- Q&A systems for internal company knowledge bases.
- Scalable and updatable retrieval-based chatbot solutions.

