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
   git clone https://github.com/yourusername/RAG-Powered-Chatbot.git
   cd RAG-Powered-Chatbot
   ```
2. **Install Dependencies**: 
 - Install the required Python libraries:
    ```bash
    pip install sentence-transformers faiss-cpu beautifulsoup4 nltk numpy
    ```
3. **Prepare the Dataset**:
 - Place your unstructured text data (e.g., capital_one_financial_report_2023.txt) in the data/ directory.
  
4. **Run the Pipeline**: 
 - Execute the RAG pipeline to query the chatbot:
    ```bash
    python src/main.py
    ```          
---

## 📚 Requirements

1. Below are the dependencies required for the project:

 - **Sentence Transformers**: sentence-transformers
 - **FAISS**: faiss-cpu
 - **BeautifulSoup (for cleaning HTML)**: beautifulsoup4
 - **NLTK**: nltk
 - **NumPy**: numpy

2. Install all dependencies with:
    ```bash
    pip install sentence-transformers faiss-cpu beautifulsoup4 nltk numpy
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
https://www.sec.gov/search-filings # FILINGS SEARCH
https://www.sec.gov/edgar/browse/?CIK=0001026214 # FREDDIE MAC SEC10K FILINGS
https://www.sec.gov/Archives/edgar/data/1026214/000102621425000116/fmcc-20250930.htm # Freddie Mac Financial Document view