import os
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document
import uuid
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    def __init__(self, collection_name="docs"):
        self.collection_name = collection_name
        # Initialize local Qdrant client
        self.client = QdrantClient("http://localhost:6333")
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # Initialize local embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.setup_collection()

    def setup_collection(self):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")

    def parse_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        text = ""
        if ext == '.pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif ext == '.docx':
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        return text

    def chunk_text(self, text, chunk_size=500):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))
        return chunks

    def index_document(self, file_path):
        text = self.parse_file(file_path)
        if not text.strip():
            return False
        
        chunks = self.chunk_text(text)
        points = []
        for chunk in chunks:
            embedding = self.model.encode(chunk).tolist()
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": chunk, "source": os.path.basename(file_path)}
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return True

    def search(self, query, top_k=3):
        query_vector = self.model.encode(query).tolist()
        # Use query_points which is the recommended modern API
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        ).points
        context = "\n".join([res.payload["text"] for res in results])
        return context

    async def query_groq(self, prompt, context, model=None):
        if model is None:
            model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        system_prompt = "You are a helpful assistant. Use the provided context to answer the user's question. If the answer is not in the context, say you don't know based on the documents."
        user_message = f"Context: {context}\n\nQuestion: {prompt}"
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                model=model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error querying Groq: {str(e)}"
