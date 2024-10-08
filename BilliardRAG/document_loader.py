from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz
import re

class DocumentLoader:
    def __init__(self, doc_path) -> None:
        """Instantiate Document Loader based on document path"""
        self.pdf_text = self.extract_pdf_text(doc_path)

    def extract_pdf_text(self, pdf_file):
        text = ""
        with fitz.open(pdf_file) as doc:
            for page in doc[3:]:
                text += page.get_text("text")  # Extract text without unintended spaces
        return text
    
    def split_by_headings(self, text):
        headings = re.split(r'\d+\.\s[A-Za-z ]+', text)  # Regex to split on numbers followed by headings
        heading_titles = re.findall(r'\d+\.\s[A-Za-z ]+', text)  # Extract the heading titles
        return list(zip(heading_titles, headings[1:]))  # Pair headings with their content
    
    def process(self, chunk_size, chunk_overlap):
        pdf_text = self.pdf_text
        sections = self.split_by_headings(pdf_text)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Create LangChain Document objects
        documents = []
        for heading, content in sections:
            # Split content into smaller chunks (LangChain works better with smaller chunks)
            split_content = text_splitter.split_text(content)
            
            for chunk in split_content:
                # Create Document object for each chunk
                doc = Document(
                    page_content=heading + " " + chunk,  # The content of the chunk
                    metadata={"heading": heading}  # Add metadata to store the section title
                )
                documents.append(doc)

        self.cleaning(documents)
        return documents
        
    def cleaning(self, docs):
        """Remove extra spaces, newline, header and footer"""
        for doc in docs:
            cleaned_str = doc.page_content.replace("\n", "")
            cleaned_str = cleaned_str.replace("Version 15.03.2016", "")
            cleaned_str = cleaned_str.replace("Version 15/03/2016 – The Rules of Play", "")
            cleaned_str = re.sub(r"Page \d+ of \d+", '', cleaned_str)
            cleaned_str = re.sub(' +', ' ', cleaned_str)
            cleaned_str = re.sub(r'\d+.\d+', '', cleaned_str)
            cleaned_str = re.sub(r'\d+. ', '', cleaned_str)
            doc.page_content = cleaned_str