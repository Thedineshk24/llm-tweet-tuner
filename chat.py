import os
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as uvicorn
import threading
from collections import defaultdict

# Thread-safe session management
class SessionManager:
    """
    Thread-safe session management for conversation history
    """
    def __init__(self, max_history_length=10):
        self._sessions = defaultdict(lambda: {
            'history': [],
            'lock': threading.Lock()
        })
        self.max_history_length = max_history_length

    def add_to_history(self, session_id: str, user_query: str, ai_response: str):
        """Add conversation turn to session history"""
        session = self._sessions[session_id]
        with session['lock']:
            session['history'].append(f"User: {user_query}")
            session['history'].append(f"AI: {ai_response}")
            
            # Trim history to max length
            if len(session['history']) > self.max_history_length * 2:
                session['history'] = session['history'][-self.max_history_length * 2:]

    def get_history(self, session_id: str) -> List[str]:
        """Retrieve conversation history for a session"""
        session = self._sessions[session_id]
        with session['lock']:
            return session['history'].copy()

    def clear_history(self, session_id: str):
        """Clear conversation history for a session"""
        session = self._sessions[session_id]
        with session['lock']:
            session['history'].clear()

# Global session manager
session_manager = SessionManager()

class DineshTweetGenerator:
    def __init__(self, 
                 pinecone_index_name="dinesh2",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        # Pinecone Setup
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index = self.pc.Index(pinecone_index_name)
        
        # Embedding Model Setup
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        
        # Load model on CPU
        self.device = torch.device("cpu")  # Ensure we use CPU
        self.model = AutoModel.from_pretrained(embedding_model).to(self.device)
        
        # Mistral LLM with LangChain
        self.mistral_llm = ChatMistralAI(
            model="mistral-large-latest", 
            api_key=os.getenv('MISTRAL_API_KEY')
        )
        
        # Personalized Dinesh-style Prompt Template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "query", "conversation_history"],
            template="""Hey there! 👋 It's Dinesh here, diving deep into the intersection of tech and human experience.

Context of our chat so far:
{conversation_history}

Relevant Background:
{context}

Your Prompt: {query}

🚀 Personal Response Mode Activated:

As someone who's spent countless nights coding, dreaming, and exploring the bleeding edge of technology, here's my unfiltered take:

I'll break this down not just as a tech response, but as a personal reflection. My thoughts will weave between technical insight and human emotion – because tech isn't just about algorithms, it's about the stories we tell and the problems we solve.

My response will include:
- A touch of technical depth
- A sprinkle of personal narrative
- An honest, slightly poetic perspective

Let's unpack this together. Here we go...

Response:"""
        )
        
        # Create LLM Chain
        self.tweet_chain = LLMChain(
            llm=self.mistral_llm, 
            prompt=self.prompt_template
        )

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for input text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Ensure input tensor is on CPU
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embedding

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve most relevant context from Pinecone based on semantic similarity
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)

        # Perform similarity search
        search_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Extract and process results
        return [
            {
                "content": result["metadata"]["content"], 
                "score": result["score"]
            } 
            for result in search_results["matches"]
        ]

    def generate_dinesh_tweet(self, query: str, contexts: List[Dict], session_id: str) -> str:
        """
        Generate a Dinesh-style tweet using retrieved contexts and the query
        """
        # Combine contexts
        context_text = "\n".join([ctx['content'] for ctx in contexts])

        # Retrieve conversation history for this session
        history_text = "\n".join(session_manager.get_history(session_id))

        try:
            # Generate tweet using LLM Chain
            tweet_response = self.tweet_chain.run(
                context=context_text,
                query=query,
                conversation_history=history_text
            )

            # Update conversation history
            session_manager.add_to_history(session_id, query, tweet_response)

            return tweet_response
        
        except Exception as e:
            error_response = f"🤖 Error generating tweet: {str(e)}"
            session_manager.add_to_history(session_id, query, error_response)
            return error_response

# FastAPI Application
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def generate_chat_response(request: ChatRequest):
    try:
        # Initialize Tweet Generator
        tweet_generator = DineshTweetGenerator()
        
        # Retrieve context
        contexts = tweet_generator.retrieve_context(request.query)
        
        # Generate Dinesh-style tweet (use a default session ID for REST endpoint)
        tweet_response = tweet_generator.generate_dinesh_tweet(
            request.query, 
            contexts,
            session_id="default_rest_session"
        )
        
        return {
            "query": request.query,
            "response": tweet_response,
            "sources": contexts
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize Tweet Generator for WebSocket connection
    tweet_generator = DineshTweetGenerator()
    
    # Generate a unique session ID for this WebSocket connection
    session_id = str(id(websocket))
    
    try:
        while True:
            # Wait for a message from the client
            query = await websocket.receive_text()
            
            try:
                # Retrieve context
                contexts = tweet_generator.retrieve_context(query)
                
                # Generate Dinesh-style tweet with session tracking
                tweet_response = tweet_generator.generate_dinesh_tweet(
                    query, 
                    contexts,
                    session_id=session_id
                )
                
                # Send response back to client
                await websocket.send_json({
                    "query": query,
                    "response": tweet_response,
                    "sources": contexts
                })
            
            except Exception as e:
                # Send error message if generation fails
                await websocket.send_json({
                    "error": str(e)
                })
    
    except WebSocketDisconnect:
        # Optional: Clean up session when connection is closed
        session_manager.clear_history(session_id)
        print(f"WebSocket connection closed for session {session_id}")

# Optional: Main function for testing
def main():
    generator = DineshTweetGenerator()
    query = "What's the latest in AI technology?"
    contexts = generator.retrieve_context(query)
    tweet = generator.generate_dinesh_tweet(query, contexts, session_id="test_session")
    print(tweet)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    main()
