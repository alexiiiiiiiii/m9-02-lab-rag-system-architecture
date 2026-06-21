import json
import chromadb
import google.generativeai as genai
import os

# Initialize Gemini
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    # Fallback to a dummy or try to proceed if we can mock it, but we need it.
    pass
genai.configure(api_key=api_key)

# We use embedding models. But chromadb has a default embedding model. We can just use chromadb's default.

def build_pipeline():
    # 1. Index
    # Initialize Chroma client
    chroma_client = chromadb.Client()
    
    # Create or get collection
    collection = chroma_client.create_collection(name="knowledge_base")
    
    # Load knowledge base
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    
    ids = []
    documents = []
    metadatas = []
    
    for item in kb_data:
        ids.append(item["id"])
        documents.append(item["text"])
        metadatas.append({"source": item["source"]})
        
    # Chunking comment:
    # If these were full documents, we would need to chunk them because embedding models 
    # have token limits, and retrieving large documents would exceed the context window 
    # of the generation model, as well as dilute the relevance of the specific information needed.
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection

def query_rag(collection, question, top_k=3):
    # 2. Query
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    
    retrieved_docs = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]
    
    context_parts = []
    for doc, meta in zip(retrieved_docs, retrieved_metadatas):
        context_parts.append(f"Source [{meta['source']}]: {doc}")
        
    context_str = "\n\n".join(context_parts)
    
    prompt = f"""You are a helpful assistant. Answer the question below based ONLY on the provided context. 
If the context does not contain the answer, say exactly "I don't know".
When you use information from the context, you MUST cite the source (e.g., [handbook.md]).

Context:
{context_str}

Question: {question}
Answer:"""
    
    # 3. Generate
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    
    return retrieved_docs, retrieved_metadatas, response.text

if __name__ == "__main__":
    collection = build_pipeline()
    
    questions = [
        "How long do I have to get a full refund?",
        "How do I reset my password?",
        "What is the company's stock price today?"
    ]
    
    print("--- STANDARD RAG RESULTS ---\n")
    for q in questions:
        docs, metas, answer = query_rag(collection, q, top_k=3)
        print(f"Question: {q}")
        print("Retrieved Sources:")
        for doc, meta in zip(docs, metas):
            print(f"- {meta['source']}: {doc[:50]}...")
        print(f"Answer: {answer}\n")
        print("-" * 40 + "\n")
        
    # Stretch goal
    print("--- STRETCH GOAL: TOP 1 RETRIEVAL ---\n")
    q_stretch = "How long do I have to get a full refund?"
    docs_stretch, metas_stretch, answer_stretch = query_rag(collection, q_stretch, top_k=1)
    print(f"Question: {q_stretch}")
    print("Retrieved Sources:")
    for doc, meta in zip(docs_stretch, metas_stretch):
         print(f"- {meta['source']}: {doc[:50]}...")
    print(f"Answer: {answer_stretch}\n")
    
    print("Trade-off between too little and too much context:")
    print("Too little context risks missing the necessary information to answer the question, while too much context can exceed the model's token limit, increase costs, and potentially distract the model with irrelevant information.")
