from qdrant_client import QdrantClient

client = QdrantClient(
    url="http://192.168.0.47:6333",
    api_key="findme-gagme-putme-inabunnyranch"
)

try:
    collections = client.get_collections()
    print("Connection successful. Collections:", collections)
except Exception as e:
    print("Connection failed:", e)
