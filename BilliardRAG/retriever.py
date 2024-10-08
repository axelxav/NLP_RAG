from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from dotenv import load_dotenv
from typing import List
import os

load_dotenv()

# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines

class Retriever:
    def __init__(self, docs, db_path):
        if not os.listdir(db_path):
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5", model_kwargs={"trust_remote_code": True}),
                collection_name="billiardRAG",
                persist_directory=db_path
            )
            query_prompt_name = "s2p_query"
            encode_kwargs={'prompt_name':query_prompt_name}
            self.vectorstore._embedding_function = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5", model_kwargs={"trust_remote_code": True}, encode_kwargs=encode_kwargs)
        else:
            self.vectorstore = Chroma(persist_directory=db_path, embedding_function=HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5", model_kwargs={"trust_remote_code": True}))
            query_prompt_name = "s2p_query"
            encode_kwargs={'prompt_name':query_prompt_name}
            self.vectorstore._embedding_function = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5", model_kwargs={"trust_remote_code": True}, encode_kwargs=encode_kwargs)
        self.llm = ChatGroq(model="llama-3.1-70b-versatile") 
    
    def retrieve(self, query):
        """Generate multiple query from llama3.1"""
        output_parser = LineListOutputParser()

        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Ensure that these alternative questions are kept as simple as possible so that the context generated from the distance-based calculation can better match what the user wants.
            Provide these alternative questions separated by newlines. Your output should be just these alternative questions without any prefixes and suffixes.
            Original question: {question}""",
        )
        
        llm_chain = QUERY_PROMPT | self.llm | output_parser

        retriever_from_llm = MultiQueryRetriever(
            retriever=self.vectorstore.as_retriever(), llm_chain=llm_chain, parser_key="lines"
        )  # "lines" is the key (attribute name) of the parsed output

        all_retrieve = retriever_from_llm.invoke(query)
        context = "\n\n".join(doc.page_content for doc in all_retrieve)
        return context
        