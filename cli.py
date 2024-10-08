from BilliardRAG.chatbot import RAGChatbot

bl_RAG = RAGChatbot(doc_path="data/billiard_rules.pdf", db_path="vectordb")

print("Welcome to Billiard Chatbot! Ask me anything about billiard!")

while True:
    user_input = input(">> ")
    if user_input == "exit":
        break
    response = bl_RAG.chat(user_input)
    print(f"<< {response}")