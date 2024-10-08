from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class Generator:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

    def generate(self, query, context):
        """Generate a response given a query and context."""
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.

        {context}

        Question: {question}

        Helpful Answer:"""
        custom_rag_prompt = PromptTemplate.from_template(template)

        rag_chain = (
            {"context": lambda x: context , "question": RunnablePassthrough()}
            | custom_rag_prompt
            | self.llm
            | StrOutputParser()
        )

        response = rag_chain.invoke(query)
        return response