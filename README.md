# Billiard RAG Chatbot

This project is a **Retrieval-Augmented Generation (RAG)** chatbot designed to answer questions related to billiard rules. It uses LangChain for retrieval and generation and integrates Groq's LLM for answering queries. The chatbot loads PDF documents containing the rules of billiards, splits them into chunks, retrieves relevant information based on user queries, and generates concise responses.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [License](#license)

## Features

- **Document Loading**: Load PDF files and split them into manageable chunks for retrieval.
- **Contextual Retrieval**: Retrieve relevant document chunks based on user queries.
- **Response Generation**: Generate responses using Groq's LLM based on retrieved context.
- **Streamlit Integration**: Provides an interactive web-based interface using Streamlit.

## Project Structure

```
├── BilliardRAG/
│   ├── __init__.py               # Package initialization
│   ├── chatbot.py                # Main RAG chatbot class
│   ├── retriever.py              # Retriever class for fetching relevant document chunks
│   ├── generator.py              # Generator class for creating responses using Groq
│   ├── document_loader.py        # Loads and processes PDF documents
│   ├── data/
│       └── billiard_rules.pdf    # Billiard rules document used by the chatbot
├── README.md                     # Project documentation
├── cli.py                        # CLI for the chatbot
├── .env                          # GROQ API KEY for the chatbot
└── app.py                        # Streamlit interface for the chatbot
```

### Key Modules:

- `chatbot.py`: The main class for managing the document loading, retrieval, and generation processes.
- `retriever.py`: Handles document vectorization and retrieval using a multiquery retrieval.
- `generator.py`: Uses a pre-trained Groq LLM to generate responses based on context.
- `document_loader.py`: Responsible for loading the PDF document and splitting it into chunks.
- `streamlit_app.py`: Provides a Streamlit-based web UI for interacting with the chatbot.

## Setup and Installation

### Prerequisites

- Python 3.9 or later
- Pip (Python package installer)

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/axelxav/NLP_RAG.git
   cd NLP_RAG
   ```

2. **Install Required Libraries**
   Install the necessary dependencies listed in `requirements.txt`. You can create a virtual environment to manage dependencies.

   ```bash
   python3 -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure the Model**
   The chatbot uses Groq's `llama-3.1-70b-versatile` model. Make sure you have the proper API key in `.env` or relevant credentials for accessing Groq (the example of this `.env` file is in `.env.example` file).

## Usage

### Running the Chatbot via Streamlit

1. **Run the Streamlit App**  
   You can run the chatbot through the web interface powered by Streamlit:

   ```bash
   streamlit run app.py
   ```

2. **Interacting with the Chatbot**
   Once the Streamlit app starts, open the provided local URL in your browser (typically `http://localhost:8501`). You will see an interactive chat interface where you can ask questions about billiard rules.

3. **Example Query**
   - User: "What happens if the cue ball is pocketed?"
   - Assistant: "If the cue ball is pocketed, it is considered a foul. The opponent gets ball-in-hand, meaning they can place the cue ball anywhere on the table for their next shot."

### Running the Chatbot via CLI

1. **Run the Streamlit App**  
   You can run the chatbot through the web interface powered by Streamlit:

   ```bash
   python cli.py
   ```

2. **Interacting with the Chatbot**
   Once the cli.py running you can interact with chatbot directly via cli.

3. **Example Query**
   - User: "What happens if the cue ball is pocketed?"
   - Assistant: "If the cue ball is pocketed, it is considered a foul. The opponent gets ball-in-hand, meaning they can place the cue ball anywhere on the table for their next shot."

## How It Works

1. **Document Loading**: The chatbot loads the `billiard_rules.pdf` file using `DocumentLoader`, which extracts the text and splits it into chunks.
2. **Retrieval**: The chunks from the loading process will then be stored into a vector database. To make these chunks into vectors, we use one of the top models in MTEB (Massive Text Embedding Benchmark) at [huggingface](https://huggingface.co/spaces/mteb/leaderboard). The embedding model we use is the stella model by [dunzhang](https://huggingface.co/dunzhang/stella_en_1.5B_v5). When a user asks a question, the chatbot uses the `Retriever` to find the most relevant chunks of text from the document. The `Retriever` utilize llama3.1 70B to create alternative queries from user input query. Each query generated by llama3.1 70B then will be used to retrieve relevant information through vector database. The relevant information retrieve from these queries will then become the context for the generation process.
3. **Generation**: The chatbot uses llama3.1 70B from Groq API to generate a concise and helpful response based on the retrieved chunks.
4. **Interface**: The chatbot interface is built using Streamlit and CLI to enable smooth interaction with the users.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
