import modal
import json
from typing import List, Dict, Any
import hashlib
from datetime import datetime

# Create Modal app
app = modal.App("lease-pinecone-system")

# Pinecone and sentence transformers image
image = modal.Image.debian_slim().pip_install([
    "pinecone-client==3.0.0",
    "sentence-transformers==2.2.2",
    "openai==1.3.0"
])

# Pinecone secret (make sure you've created this)
pinecone_secret = modal.Secret.from_name("pinecone-secret")

@app.function(
    image=image,
    secrets=[pinecone_secret],
    timeout=300
)
def store_in_pinecone(
    document_id: str,
    processed_data: Dict[str, Any],
    chunks_with_pages: List[Dict[str, Any]],
    filename: str
):
    """Store processed document chunks in Pinecone with page references"""
    import pinecone
    from sentence_transformers import SentenceTransformer
    import os
    
    try:
        # Initialize Pinecone
        api_key = os.environ["PINECONE_API_KEY"]
        pc = pinecone.Pinecone(api_key=api_key)
        
        # Connect to index
        index_name = "lease-documents"
        index = pc.Index(index_name)
        
        # Initialize embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        vectors_to_upsert = []
        
        # Process each chunk
        for i, chunk_data in enumerate(chunks_with_pages):
            chunk_text = chunk_data.get('text', '')
            page_number = chunk_data.get('page_number', 1)
            
            if not chunk_text.strip():
                continue
                
            # Generate embedding
            embedding = model.encode(chunk_text).tolist()
            
            # Create unique vector ID
            vector_id = f"{document_id}_chunk_{i}_{hashlib.md5(chunk_text.encode()).hexdigest()[:8]}"
            
            # Prepare metadata
            metadata = {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
                "page_number": page_number,
                "text": chunk_text[:1000],  # Truncate for metadata
                "chunk_type": chunk_data.get('chunk_type', 'content'),
                "created_at": datetime.now().isoformat(),
                "processed_data": json.dumps(processed_data)[:500]  # Truncate
            }
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })
        
        # Upsert vectors in batches
        batch_size = 100
        total_vectors = len(vectors_to_upsert)
        
        for i in range(0, total_vectors, batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"Upserted batch {i//batch_size + 1}/{(total_vectors + batch_size - 1)//batch_size}")
        
        return {
            "status": "success",
            "document_id": document_id,
            "chunks_stored": total_vectors,
            "filename": filename,
            "index_name": index_name
        }
        
    except Exception as e:
        print(f"Error storing in Pinecone: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "document_id": document_id
        }

@app.function(
    image=image,
    secrets=[pinecone_secret],
    timeout=300
)
def query_documents(
    question: str,
    document_filter: str = None,
    top_k: int = 5,
    include_page_refs: bool = True
):
    """Query documents in Pinecone and return relevant chunks with page references"""
    import pinecone
    from sentence_transformers import SentenceTransformer
    import os
    
    try:
        # Initialize Pinecone
        api_key = os.environ["PINECONE_API_KEY"]
        pc = pinecone.Pinecone(api_key=api_key)
        
        # Connect to index
        index_name = "lease-documents"
        index = pc.Index(index_name)
        
        # Initialize embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate query embedding
        query_embedding = model.encode(question).tolist()
        
        # Prepare filter
        filter_dict = {}
        if document_filter:
            filter_dict["filename"] = {"$eq": document_filter}
        
        # Query Pinecone
        query_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        # Process results
        relevant_chunks = []
        context_metadata = {
            "sources": [],
            "page_references": [],
            "documents": set()
        }
        
        for match in query_results.matches:
            metadata = match.metadata
            
            chunk_info = {
                "text": metadata.get("text", ""),
                "page_number": metadata.get("page_number", 1),
                "filename": metadata.get("filename", ""),
                "document_id": metadata.get("document_id", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "score": float(match.score)
            }
            
            relevant_chunks.append(chunk_info)
            
            # Collect metadata
            context_metadata["sources"].append({
                "filename": metadata.get("filename", ""),
                "page": metadata.get("page_number", 1),
                "score": float(match.score)
            })
            
            context_metadata["page_references"].append({
                "page": metadata.get("page_number", 1),
                "filename": metadata.get("filename", ""),
                "text_preview": metadata.get("text", "")[:100] + "..."
            })
            
            context_metadata["documents"].add(metadata.get("filename", ""))
        
        # Convert set to list for JSON serialization
        context_metadata["documents"] = list(context_metadata["documents"])
        
        return {
            "status": "success",
            "relevant_chunks": relevant_chunks,
            "context_metadata": context_metadata,
            "query": question,
            "total_results": len(relevant_chunks)
        }
        
    except Exception as e:
        print(f"Error querying documents: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "query": question
        }

@app.function(
    image=image,
    secrets=[pinecone_secret],
    timeout=300
)
def generate_answer(
    question: str,
    relevant_chunks: List[Dict[str, Any]],
    context_metadata: Dict[str, Any]
):
    """Generate AI answer using OpenAI with page references"""
    import openai
    import os
    
    try:
        # Initialize OpenAI (you'll need to add this secret too)
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Prepare context from chunks
        context_text = ""
        page_refs = []
        
        for chunk in relevant_chunks:
            context_text += f"\n[Page {chunk['page_number']} of {chunk['filename']}]\n{chunk['text']}\n"
            page_refs.append({
                "page": chunk['page_number'],
                "filename": chunk['filename'],
                "relevance_score": chunk['score']
            })
        
        # Create prompt
        prompt = f"""Based on the following lease document excerpts, answer the question: "{question}"

Context from lease documents:
{context_text}

Please provide a clear, accurate answer based on the information provided. If the answer references specific information, mention the page number and document name.

Question: {question}"""

        # Generate response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes lease documents. Always reference page numbers when citing specific information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        return {
            "status": "success",
            "answer": answer,
            "confidence": "high" if len(relevant_chunks) > 2 else "medium",
            "sources": context_metadata.get("sources", []),
            "page_references": page_refs,
            "question": question
        }
        
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "question": question
        }

# Web endpoints
@app.function(image=image, secrets=[pinecone_secret])
@modal.web_endpoint(method="POST")
def store_endpoint(item: Dict[str, Any]):
    """Web endpoint for storing documents"""
    return store_in_pinecone.remote(
        document_id=item["document_id"],
        processed_data=item["processed_data"],
        chunks_with_pages=item["chunks_with_pages"],
        filename=item["filename"]
    )

@app.function(image=image, secrets=[pinecone_secret])
@modal.web_endpoint(method="POST")
def query_endpoint(item: Dict[str, Any]):
    """Web endpoint for querying documents"""
    return query_documents.remote(
        question=item["question"],
        document_filter=item.get("document_filter"),
        top_k=item.get("top_k", 5),
        include_page_refs=item.get("include_page_refs", True)
    )

@app.function(image=image, secrets=[pinecone_secret])
@modal.web_endpoint(method="POST")
def answer_endpoint(item: Dict[str, Any]):
    """Web endpoint for generating answers"""
    return generate_answer.remote(
        question=item["question"],
        relevant_chunks=item["relevant_chunks"],
        context_metadata=item["context_metadata"]
    )
