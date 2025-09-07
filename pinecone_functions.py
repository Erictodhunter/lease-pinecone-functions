import modal
import json
from typing import List, Dict, Any
import hashlib
from datetime import datetime

app = modal.App("lease-pinecone-system")

image = modal.Image.debian_slim().pip_install([
    "pinecone-client==3.0.0",
    "sentence-transformers==2.2.2",
    "fastapi"
])

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
    import pinecone
    from sentence_transformers import SentenceTransformer
    import os
    
    try:
        api_key = os.environ["PINECONE_API_KEY"]
        pc = pinecone.Pinecone(api_key=api_key)
        
        index_name = "lease-documents"
        index = pc.Index(index_name)
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        vectors_to_upsert = []
        
        for i, chunk_data in enumerate(chunks_with_pages):
            chunk_text = chunk_data.get('text', '')
            page_number = chunk_data.get('page_number', 1)
            
            if not chunk_text.strip():
                continue
                
            embedding = model.encode(chunk_text).tolist()
            
            vector_id = f"{document_id}_chunk_{i}_{hashlib.md5(chunk_text.encode()).hexdigest()[:8]}"
            
            metadata = {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
                "page_number": page_number,
                "text": chunk_text[:1000],
                "chunk_type": chunk_data.get('chunk_type', 'content'),
                "created_at": datetime.now().isoformat(),
                "processed_data": json.dumps(processed_data)[:500]
            }
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })
        
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
    import pinecone
    from sentence_transformers import SentenceTransformer
    import os
    
    try:
        api_key = os.environ["PINECONE_API_KEY"]
        pc = pinecone.Pinecone(api_key=api_key)
        
        index_name = "lease-documents"
        index = pc.Index(index_name)
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        query_embedding = model.encode(question).tolist()
        
        filter_dict = {}
        if document_filter:
            filter_dict["filename"] = {"$eq": document_filter}
        
        query_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
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
def generate_simple_answer(
    question: str,
    relevant_chunks: List[Dict[str, Any]],
    context_metadata: Dict[str, Any]
):
    try:
        best_chunks = sorted(relevant_chunks, key=lambda x: x['score'], reverse=True)[:3]
        
        answer_parts = []
        page_refs = []
        
        for chunk in best_chunks:
            if chunk['score'] > 0.7:
                answer_parts.append(f"From page {chunk['page_number']}: {chunk['text'][:200]}...")
                page_refs.append({
                    "page": chunk['page_number'],
                    "filename": chunk['filename'],
                    "relevance_score": chunk['score']
                })
        
        if not answer_parts:
            answer = "No highly relevant information found for this question."
            confidence = "low"
        else:
            answer = "\n\n".join(answer_parts)
            confidence = "high" if len(answer_parts) > 1 else "medium"
        
        return {
            "status": "success",
            "answer": answer,
            "confidence": confidence,
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

@app.function(image=image, secrets=[pinecone_secret])
@modal.fastapi_endpoint(method="POST")
def store_endpoint(item: Dict[str, Any]):
    return store_in_pinecone.remote(
        document_id=item["document_id"],
        processed_data=item["processed_data"],
        chunks_with_pages=item["chunks_with_pages"],
        filename=item["filename"]
    )

@app.function(image=image, secrets=[pinecone_secret])
@modal.fastapi_endpoint(method="POST")
def query_endpoint(item: Dict[str, Any]):
    return query_documents.remote(
        question=item["question"],
        document_filter=item.get("document_filter"),
        top_k=item.get("top_k", 5),
        include_page_refs=item.get("include_page_refs", True)
    )

@app.function(image=image, secrets=[pinecone_secret])
@modal.fastapi_endpoint(method="POST")
def answer_endpoint(item: Dict[str, Any]):
    return generate_simple_answer.remote(
        question=item["question"],
        relevant_chunks=item["relevant_chunks"],
        context_metadata=item["context_metadata"]
    )
