"""Qdrant Vector Index Module.

Replaces the in-process FAISS index (THESIS_TRACKER §3.6). Qdrant runs as a
container beside the service, so persistence, real deletes and payload
pre-filtering come from the store instead of hand-rolled code.

Keeps the FAISSIndex method shape so the API layer and build scripts change
only their constructor call.
"""
import logging
import uuid

import numpy as np
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

# Payload fields filtered on at search time (DATASET_SPEC §3).
INDEXED_PAYLOAD_FIELDS = ("category", "brand", "part_number", "is_distractor")


class QdrantIndex:
    """Qdrant-backed vector index for image and text embeddings.

    One point per embedding; several points may share a ``product_id`` (a part
    photographed from several angles). The product id lives in the payload.
    """

    def __init__(
        self,
        dimension: int,
        collection: str,
        url: str = "http://localhost:6333",
        metric: str = "l2",
        hnsw_m: int = 32,
        hnsw_ef_construct: int = 100,
        hnsw_ef_search: int = 64,
        client: QdrantClient | None = None,
    ):
        """Initialize the index.

        Args:
            dimension: Embedding dimension.
            collection: Qdrant collection name.
            url: Qdrant URL, or ":memory:" for an in-process store (tests).
            metric: Score scale returned by search(). The stored distance is
                always cosine (embeddings are L2-normalised); this only maps
                the score so service thresholds keep their meaning:
                'l2' -> 1/(1+d²) as FAISS IndexFlatL2 gave, 'inner_product'
                -> (cos+1)/2, 'cosine' -> raw cosine.
            hnsw_m, hnsw_ef_construct, hnsw_ef_search: HNSW parameters for the
                service path. Ignored by exact searches.
            client: Pre-built client, overrides ``url``.
        """
        if metric not in ("l2", "inner_product", "cosine"):
            raise ValueError(f"Unknown metric: {metric}")
        self.dimension = dimension
        self.collection = collection
        self.metric = metric
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        self.hnsw_ef_search = hnsw_ef_search
        self.client = client or QdrantClient(location=url)
        logger.info(f"Initializing Qdrant index (collection={collection}, dim={dimension}, url={url})")

    def create_index(self):
        """Create the collection and its payload indexes if missing."""
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct),
        )
        for field in INDEXED_PAYLOAD_FIELDS:
            schema = models.PayloadSchemaType.BOOL if field == "is_distractor" else models.PayloadSchemaType.KEYWORD
            self.client.create_payload_index(self.collection, field, field_schema=schema)
        logger.info(f"Created Qdrant collection '{self.collection}'")

    def load_index(self) -> bool:
        """Ensure the collection exists. Returns True if it already holds vectors."""
        self.create_index()
        return self.count() > 0

    def save_index(self):
        """No-op: Qdrant persists every write. Kept for API compatibility."""

    def train(self, embeddings: np.ndarray):
        """No-op: nothing to train. Kept for API compatibility."""

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        product_ids: list,
        payloads: list[dict] | None = None,
    ) -> int:
        """Add embeddings, one point each.

        Args:
            embeddings: (n, dimension) array.
            product_ids: Product id per row: int in the service, part-number
                string in the research harness. Stored as given.
            payloads: Optional extra payload per row (category, brand, ...).

        Returns:
            Number of points added.
        """
        self.create_index()
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(embeddings) != len(product_ids):
            raise ValueError("embeddings and product_ids differ in length")
        payloads = payloads or [{} for _ in product_ids]
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload={**extra, "product_id": pid},
            )
            for vec, pid, extra in zip(embeddings, product_ids, payloads)
        ]
        # Batches keep each request under Qdrant's payload size limit.
        for start in range(0, len(points), 256):
            self.client.upsert(self.collection, points[start:start + 256], wait=True)
        logger.info(f"Added {len(points)} embeddings to '{self.collection}'")
        return len(points)

    def _score(self, cosine: float) -> float:
        if self.metric == "l2":
            # Unit vectors: squared L2 = 2 - 2cos. Same value FAISS returned.
            return float(1.0 / (1.0 + max(0.0, 2.0 - 2.0 * cosine)))
        if self.metric == "inner_product":
            return float(max(0.0, (cosine + 1.0) / 2.0))
        return float(cosine)

    @staticmethod
    def _filter(filters: dict | None) -> models.Filter | None:
        if not filters:
            return None
        return models.Filter(must=[
            models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in filters.items()
        ])

    def _params(self, exact: bool) -> models.SearchParams:
        return models.SearchParams(exact=exact, hnsw_ef=None if exact else self.hnsw_ef_search)

    def _hits(self, points) -> list[tuple]:
        return [(p.payload["product_id"], self._score(p.score)) for p in points]

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filters: dict | None = None,
        exact: bool = False,
    ) -> list[tuple]:
        """Search for similar embeddings.

        Args:
            query_embedding: Query vector.
            k: Number of points to return.
            filters: Payload equality pre-filter, e.g. {"category": "oil_filter"}.
            exact: Brute-force search. Research runs pass True so metrics
                measure the embedding, not HNSW recall.

        Returns:
            List of (product_id, similarity), best first. A product can appear
            more than once if it has several images.
        """
        if not self.client.collection_exists(self.collection):
            logger.warning("Index is empty, cannot search")
            return []
        points = self.client.query_points(
            self.collection,
            query=np.asarray(query_embedding, dtype=np.float32).ravel().tolist(),
            limit=k,
            query_filter=self._filter(filters),
            search_params=self._params(exact),
        ).points
        return self._hits(points)

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        filters: dict | None = None,
        exact: bool = False,
    ) -> list[list[tuple]]:
        """Batch form of search(): one result list per query row."""
        if not self.client.collection_exists(self.collection):
            return [[] for _ in range(len(query_embeddings))]
        requests = [
            models.QueryRequest(
                query=np.asarray(q, dtype=np.float32).tolist(),
                limit=k,
                filter=self._filter(filters),
                params=self._params(exact),
                with_payload=True,
            )
            for q in query_embeddings
        ]
        responses = self.client.query_batch_points(self.collection, requests=requests)
        return [self._hits(r.points) for r in responses]

    def remove_product(self, product_id) -> bool:
        """Delete every vector of a product. Real delete, unlike the FAISS mapping-only one.

        Returns:
            True if anything was removed.
        """
        if not self.client.collection_exists(self.collection):
            return False
        selector = self._filter({"product_id": product_id})
        if self.client.count(self.collection, count_filter=selector, exact=True).count == 0:
            return False
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=selector), wait=True)
        logger.info(f"Removed product {product_id} from '{self.collection}'")
        return True

    def search_with_metadata(self, query_embedding: np.ndarray, k: int = 10, filters: dict | None = None) -> list[dict]:
        """Search, returning each hit's payload beside its score."""
        if not self.client.collection_exists(self.collection):
            return []
        points = self.client.query_points(
            self.collection,
            query=np.asarray(query_embedding, dtype=np.float32).ravel().tolist(),
            limit=k,
            query_filter=self._filter(filters),
            search_params=self._params(False),
        ).points
        return [
            {**p.payload, "similarity": self._score(p.score)}
            for p in points
        ]

    def get_stats(self) -> dict:
        """Index statistics for the /index/stats endpoint."""
        if not self.client.collection_exists(self.collection) or self.count() == 0:
            return {"status": "empty", "total_vectors": 0, "unique_products": 0}
        product_ids = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=1000, offset=offset, with_payload=["product_id"], with_vectors=False
            )
            product_ids.update(p.payload["product_id"] for p in points)
            if offset is None:
                break
        return {
            "status": "ready",
            "total_vectors": self.count(),
            "unique_products": len(product_ids),
            "dimension": self.dimension,
            "collection": self.collection,
        }

    def clear(self):
        """Drop and recreate the collection."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.create_index()
        logger.info(f"Cleared '{self.collection}'")
