import json
from pathlib import Path
from backend.data_types import Document, DocumentMetadata, DocumentStatus
from dataclasses import asdict


def doc_from_dict(data: dict) -> Document:
    meta = data.get("metadata", {})
    if isinstance(meta, dict):
        data = {**data, "metadata": DocumentMetadata(**meta)}
    return Document(**data)

class DocumentRegistry:
    def __init__(self,storage_dir: Path) ->None:
        self._path = storage_dir / "document_registry.json"
        self._docs = self._load()
    def _load(self):
        if self._path.exists():
            with open(self._path) as f:
                return json.load(f)
        return {}
    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self._docs, f, indent=2, default=str)

    def insert(self, doc: Document):
        self._docs[doc.id] = asdict(doc)
        self._save()
    def get(self,doc_id):
        data =self._docs.get(doc_id)
        if data:
            return doc_from_dict(data)
        return None
    def list_all(self):
        return [doc_from_dict(d) for d in self._docs.values()]
    def delete(self, doc_id):
        if doc_id in self._docs:
            del self._docs[doc_id]
            self._save()
            return True
        return False
    def update_status(self, doc_id, status: DocumentStatus, chunk_count=0):
        if doc_id not in self._docs:
            return

        self._docs[doc_id]["status"] = status.value
        if chunk_count:
            self._docs[doc_id]["chunk_count"] = chunk_count
        self._save()
