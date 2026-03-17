import os
from typing import List, Dict, Any, Optional
import logging
import requests
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient


class RAGEngine:
    """
    Retrieval-Augmented Generation (RAG) engine for document Q&A.

    Handles document search, embedding generation, and LLM-based answer generation
    using Azure OpenAI and Azure Cognitive Search.
    """

    def __init__(self):
        """Initialize the RAG engine with Azure services."""
        self.logger = logging.getLogger(__name__)

        # Load configuration from environment
        self._load_config()

        # Initialize clients
        self._init_clients()

    def _load_config(self):
        """Load and validate environment variables."""
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        self.api_key = os.getenv("AZURE_OPENAI_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.embedding_model = os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME")
        self.chat_model = os.getenv("CHAT_MODEL_DEPLOYMENT_NAME")
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT") or self.openai_endpoint
        self.search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "documents-index")
        self.storage_connection_string = os.getenv("STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("BLOB_CONTAINER_NAME")

        # Validate required variables
        required = {
            "AZURE_OPENAI_ENDPOINT": self.openai_endpoint,
            "AZURE_OPENAI_KEY": self.api_key,
            "AZURE_OPENAI_API_VERSION": self.api_version,
            "EMBEDDING_MODEL_DEPLOYMENT_NAME": self.embedding_model,
            "CHAT_MODEL_DEPLOYMENT_NAME": self.chat_model,
            "STORAGE_CONNECTION_STRING": self.storage_connection_string,
            "BLOB_CONTAINER_NAME": self.container_name,
        }

        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    def _init_clients(self):
        """Initialize Azure clients."""
        # Search client (optional)
        self.search_client = None
        self.search_index_client = None
        if os.getenv("AZURE_SEARCH_KEY"):
            self.search_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.search_index_name,
                credential=self._azure_search_credential(),
            )
            # Create index client for management
            try:
                from azure.search.documents.indexes import SearchIndexClient
                self.search_index_client = SearchIndexClient(
                    endpoint=self.search_endpoint,
                    credential=self._azure_search_credential(),
                )
                self._ensure_search_index(self.search_index_name)
            except Exception as e:
                self.logger.warning("Failed to initialize search index client: %s", e)
        else:
            self.logger.warning("AZURE_SEARCH_KEY not set; search disabled.")

        # Blob storage client
        self.blob_service = BlobServiceClient.from_connection_string(self.storage_connection_string)

    def _azure_search_credential(self):
        """Create Azure Key Credential for Cognitive Search."""
        from azure.core.credentials import AzureKeyCredential
        key = os.getenv("AZURE_SEARCH_KEY")
        if not key:
            raise ValueError("AZURE_SEARCH_KEY is required for SearchClient")
        return AzureKeyCredential(key)

    def _openai_url(self, path: str) -> str:
        """Build full Azure OpenAI API URL."""
        return f"{self.openai_endpoint}/openai{path}?api-version={self.api_version}"

    def _openai_headers(self) -> Dict[str, str]:
        """Get headers for Azure OpenAI API calls."""
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    def _ensure_search_index(self, index_name: str) -> None:
        """Create search index if it doesn't exist."""
        try:
            from azure.search.documents.indexes.models import (
                SearchIndex, SearchableField, SimpleField, SearchFieldDataType
            )
        except ImportError:
            self.logger.warning("azure-search-documents models not available; cannot create index.")
            return

        try:
            self.search_index_client.get_index(index_name)
            return  # Index exists
        except Exception:
            pass  # Create it

        # Define minimal index schema
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SimpleField(name="metadata", type=SearchFieldDataType.String, filterable=False, searchable=False),
        ]

        index = SearchIndex(name=index_name, fields=fields)
        try:
            self.search_index_client.create_or_update_index(index)
            self.logger.info("Created search index '%s'", index_name)
        except Exception as e:
            self.logger.warning("Could not create search index '%s': %s", index_name, e)

    def _get_index_document_count(self) -> Optional[int]:
        """Get document count in search index."""
        if not self.search_client:
            return None
        try:
            stats = self.search_client.get_index_statistics()
            return stats.document_count
        except Exception as e:
            self.logger.debug("Unable to get index stats: %s", e)
            return None

    def _search_with_select_fallback(self, **kwargs):
        """Search with fallback if select fields don't exist."""
        from azure.core.exceptions import HttpResponseError

        try:
            return list(self.search_client.search(**kwargs))
        except HttpResponseError as e:
            if "Could not find a property named" in str(e) or "Invalid expression" in str(e):
                self.logger.warning("Retrying search without select fields: %s", e)
                kwargs.pop("select", None)
                return list(self.search_client.search(**kwargs))
            raise

    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using embeddings and keyword search."""
        # Generate query embedding
        try:
            query_vector = self._get_embedding(query)
        except Exception as e:
            self.logger.error("Embedding generation failed: %s", e)
            raise

        if not self.search_client:
            self.logger.warning("Search disabled; no documents available.")
            return []

        # Search with fallback for missing fields
        try:
            results = self._search_with_select_fallback(
                search_text=query,
                top=top_k,
                include_total_count=True,
                select=["id", "content", "metadata", "title"],
            )
        except Exception as e:
            from azure.core.exceptions import ResourceNotFoundError
            if isinstance(e, ResourceNotFoundError):
                self.logger.warning("Search index not found; returning empty results.")
                return []
            self.logger.error("Search failed: %s", e)
            raise

        # Format results
        docs = [
            {
                "id": r.get("id"),
                "content": str(r.get("content") or ""),
                "title": r.get("metadata", {}).get("filename", "Unknown"),
                "score": r.get("@search.score"),
                "metadata": r.get("metadata", {}),
            }
            for r in results
        ]

        # Fallback to wildcard if no matches
        if not docs:
            indexed_count = self._get_index_document_count()
            if indexed_count and indexed_count > 0:
                self.logger.warning("No matches for query; using wildcard fallback.")
                try:
                    fallback_results = self._search_with_select_fallback(
                        search_text="*",
                        top=top_k,
                        select=["id", "content", "metadata", "title"],
                    )
                    docs = [
                        {
                            "id": r.get("id"),
                            "content": str(r.get("content") or ""),
                            "title": r.get("metadata", {}).get("filename", "Unknown"),
                            "score": r.get("@search.score"),
                            "metadata": r.get("metadata", {}),
                        }
                        for r in fallback_results
                    ]
                    if docs:
                        self.logger.info("Fallback retrieved %d documents.", len(docs))
                except Exception as e:
                    self.logger.debug("Fallback search failed: %s", e)
            else:
                self.logger.warning("Index empty or unavailable.")

        return docs

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        url = self._openai_url(f"/deployments/{self.embedding_model}/embeddings")
        response = requests.post(
            url,
            headers=self._openai_headers(),
            json={"input": [text]},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def generate_answer(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Generate RAG-based answer for query."""
        documents = self.search_documents(query)

        if not documents:
            count = self._get_index_document_count()
            if count is None:
                msg = "No search configured."
            elif count == 0:
                msg = "Knowledge base is empty."
            else:
                msg = f"No relevant documents found ({count} total)."
            return {"answer": f"Sorry, I couldn't find any relevant information. {msg}", "sources": []}

        # Build context from documents
        context = "\n\n".join(f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(documents))

        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Cite sources as [Document X]."},
        ]
        if chat_history:
            messages.extend(chat_history[-10:])  # Keep recent history
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

        # Generate response
        response = self._create_chat_completion(messages)
        return {
            "answer": response["choices"][0]["message"]["content"],
            "sources": documents,
        }

    def _create_chat_completion(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call Azure OpenAI chat completion API."""
        url = self._openai_url(f"/deployments/{self.chat_model}/chat/completions")
        self.logger.info("Calling OpenAI: %s", url)
        response = requests.post(
            url,
            headers=self._openai_headers(),
            json={"messages": messages, "temperature": 0.3, "max_tokens": 800},
            timeout=60,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            self.logger.error("OpenAI API error: %s, response: %s", e, response.text)
            raise
        return response.json()

    def upload_document(self, file_content: bytes, filename: str) -> str:
        """Upload document to blob storage."""
        if not self.container_name:
            raise ValueError("BLOB_CONTAINER_NAME not set")

        # Ensure container exists
        container_client = self.blob_service.get_container_client(self.container_name)
        try:
            container_client.create_container()
        except Exception:
            pass  # Already exists

        # Upload file
        blob_client = container_client.get_blob_client(blob=filename)
        blob_client.upload_blob(file_content, overwrite=True)
        return blob_client.url

