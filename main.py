# import json
# import os
# import uuid
# from typing import List, Dict

# from pinecone import Pinecone, ServerlessSpec
# from langchain_openai import OpenAIEmbeddings
# from langchain_core.documents import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv

# class SocialMediaPostEmbedder:
#     def __init__(self, 
#                  openai_api_key: str, 
#                  pinecone_api_key: str,
#                  pinecone_index_name: str = 'dinesh',  # Your index name
#                  embedding_model: str = 'text-embedding-ada-002'):
#         """
#         Initialize the Social Media Post Embedder with Pinecone integration
        
#         :param openai_api_key: Your OpenAI API key
#         :param pinecone_api_key: Your Pinecone API key
#         :param pinecone_index_name: Name of the Pinecone index
#         :param embedding_model: OpenAI embedding model to use
#         """
#         # Initialize Pinecone
#         self.pc = Pinecone(api_key=pinecone_api_key)
        
#         # Get the index
#         self.index = self.pc.Index(pinecone_index_name)
        
#         # Initialize OpenAI Embeddings (using new import)
#         self.embeddings = OpenAIEmbeddings(
#             openai_api_key=openai_api_key,
#             model=embedding_model
#         )
        
#         # Text splitter
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500,  # Smaller chunk size for social media posts
#             chunk_overlap=100  # Some overlap to maintain context
#         )
    
#     def load_json(self, file_path: str) -> List[Document]:
#         """
#         Load JSON file and convert to LangChain Documents
        
#         :param file_path: Path to the JSON file
#         :return: List of Document objects
#         """
#         with open(file_path, 'r', encoding='utf-8') as file:
#             data = json.load(file)
        
#         documents = []
#         for post in data:
#             # Extract relevant information for embedding
#             content = post.get('Content', '')
#             date = post.get('Date', '')
#             post_type = post.get('Post Type', '')
            
#             # Create a structured document
#             document_text = f"Date: {date}\nPost Type: {post_type}\nContent: {content}"
            
#             documents.append(Document(
#                 page_content=document_text,
#                 metadata={
#                     'date': date,
#                     'post_type': post_type,
#                     'original_content': content
#                 }
#             ))
        
#         return documents
    
#     def split_documents(self, documents: List[Document]) -> List[Document]:
#         """
#         Split documents into smaller chunks
        
#         :param documents: Original documents
#         :return: List of split documents
#         """
#         return self.text_splitter.split_documents(documents)
    
#     def upsert_to_pinecone(self, documents: List[Document], embeddings: List[List[float]]):
#         """
#         Upsert embeddings to Pinecone
        
#         :param documents: List of documents
#         :param embeddings: List of vector embeddings
#         """
#         # Prepare vectors for Pinecone
#         vectors = []
#         for doc, embedding in zip(documents, embeddings):
#             # Generate a unique ID for each vector
#             vector_id = str(uuid.uuid4())
            
#             # Convert metadata to string values for Pinecone compatibility
#             metadata = {
#                 str(k): str(v) for k, v in doc.metadata.items()
#             }
            
#             vectors.append({
#                 'id': vector_id,
#                 'values': embedding,
#                 'metadata': metadata
#             })
        
#         # Upsert vectors to Pinecone
#         try:
#             # Upsert in batches to handle large datasets
#             batch_size = 100
#             for i in range(0, len(vectors), batch_size):
#                 batch = vectors[i:i+batch_size]
#                 self.index.upsert(batch)
#                 print(f"Upserted batch {i//batch_size + 1}: {len(batch)} vectors")
            
#             print(f"Successfully upserted {len(vectors)} total vectors to Pinecone")
#         except Exception as e:
#             print(f"Error upserting to Pinecone: {e}")
    
#     def process_posts_to_embeddings(self, file_path: str):
#         """
#         Full process of converting social media posts to embeddings and storing in Pinecone
        
#         :param file_path: Path to the JSON file
#         :return: Dictionary containing original documents and their embeddings
#         """
#         # Load social media posts
#         documents = self.load_json(file_path)
        
#         # Split documents
#         split_documents = self.split_documents(documents)
        
#         # Create embeddings
#         embeddings = self.embeddings.embed_documents(
#             [doc.page_content for doc in split_documents]
#         )
        
#         # Upsert to Pinecone
#         self.upsert_to_pinecone(split_documents, embeddings)
        
#         return {
#             'documents': split_documents,
#             'embeddings': embeddings
#         }

# def main():
#     # Load environment variables
#     load_dotenv()
    
#     # Retrieve API keys from environment variables
#     OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
#     PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    
#     # Create embedder instance
#     post_embedder = SocialMediaPostEmbedder(
#         openai_api_key=OPENAI_API_KEY,
#         pinecone_api_key=PINECONE_API_KEY
#     )
    
#     try:
#         # Process JSON file and upsert to Pinecone
#         result = post_embedder.process_posts_to_embeddings('./Thedineshk24.json')
        
#         # Print summary information
#         print(f"Total documents processed: {len(result['documents'])}")
#         print(f"Embedding dimension: {len(result['embeddings'][0])}")
#         print(f"Total vectors upserted: {len(result['embeddings'])}")
#         print(result['documents'])
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()



from transformers import AutoTokenizer, AutoModel
import torch
import json
import uuid
from typing import List, Dict
from dotenv import load_dotenv
import os

class SocialMediaPostEmbedder:
    def __init__(self, 
                 pinecone_api_key: str,
                 pinecone_index_name: str = 'dinesh2',  # Your index name
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the Social Media Post Embedder with Pinecone integration and Mistral
        
        :param pinecone_api_key: Your Pinecone API key
        :param pinecone_index_name: Name of the Pinecone index
        :param model_name: Hugging Face model for embeddings
        """
        # Initialize Pinecone
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        
        # Load Mistral model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Mistral.
        
        :param texts: List of strings to encode
        :return: List of embeddings
        """
        embeddings = []
        for text in texts:
            # Tokenize input text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # Mean pooling
            embeddings.append(embedding)
        return embeddings
    
    def process_posts_to_embeddings(self, file_path: str):
        """
        Full process of converting social media posts to embeddings and storing in Pinecone
        
        :param file_path: Path to the JSON file
        :return: Dictionary containing original documents and their embeddings
        """
        # Load social media posts
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        texts = [post.get('Content', '') for post in data]
        embeddings = self.encode_texts(texts)
        
        # Upsert to Pinecone
        vectors = [
            {
                'id': str(uuid.uuid4()),
                'values': embedding,
                'metadata': {"content": text}
            }
            for text, embedding in zip(texts, embeddings)
        ]
        
        self.index.upsert(vectors)
        return {"texts": texts, "embeddings": embeddings}

def main():
    # Load environment variables
    load_dotenv()
    
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    
    # Initialize the embedder
    embedder = SocialMediaPostEmbedder(
        pinecone_api_key=PINECONE_API_KEY
    )
    
    # Process JSON file
    result = embedder.process_posts_to_embeddings('./Thedineshk24.json')
    
    print(f"Processed {len(result['texts'])} posts and stored embeddings.")

if __name__ == "__main__":
    main()
