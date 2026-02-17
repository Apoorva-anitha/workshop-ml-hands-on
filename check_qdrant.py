from qdrant_client import QdrantClient
client = QdrantClient("http://localhost:6333")
print(f"Client: {client}")
print(f"Has search: {hasattr(client, 'search')}")
print(f"Methods: {[m for m in dir(client) if not m.startswith('_')]}")
