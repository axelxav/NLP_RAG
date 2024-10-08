from BilliardRAG.document_loader import DocumentLoader
from BilliardRAG.retriever import Retriever
from BilliardRAG.generator import Generator

class RAGChatbot:
    def __init__(self, doc_path, db_path):
        self.loader = DocumentLoader(doc_path)
        self.docs = self.loader.process(1000, 200)
        print("Reading Database...")
        self.retriever = Retriever(self.docs, db_path)
        print("Database Loaded...")
        self.generator = Generator()

    def chat(self, query):
        """Handle the query by retrieving context and generating a response."""
        context = self.retriever.retrieve(query)
        response = self.generator.generate(query, context)
        return response
