"""Catalog Client Module.

This module provides HTTP client for communicating with the main API
to search the product catalog.
"""
import httpx
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CatalogClient:
    """HTTP client for main API catalog operations."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 5.0):
        """Initialize the catalog client.

        Args:
            base_url: Base URL of the main API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._category_cache: Dict[str, int] = {}  # name → category_id

        logger.info(f"CatalogClient initialized with base URL: {self.base_url}")

    # ------------------------------------------------------------------
    # Category cache (Option B: ML service resolves names → IDs)
    # ------------------------------------------------------------------
    async def _load_categories(self):
        """Fetch categories from backend and build name→ID cache."""
        if self._category_cache:
            return
        url = f"{self.base_url}/category/"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    for cat in response.json():
                        self._category_cache[cat["name"].lower()] = cat["categoryID"]
                    logger.info(f"Loaded {len(self._category_cache)} categories")
            except Exception as e:
                logger.warning(f"Could not load categories: {e}")

    def get_category_id(self, category_name: str) -> Optional[int]:
        """Resolve category name to ID (case-insensitive).

        Args:
            category_name: Category name from YOLO detection

        Returns:
            category_id (int) or None if unknown
        """
        return self._category_cache.get(category_name.lower())

    # ------------------------------------------------------------------
    # Product search
    # ------------------------------------------------------------------
    async def search_by_category(
        self,
        category_id: int,
        skip: int = 0,
        limit: int = 20,
    ) -> List[Dict]:
        """Search products by category ID.

        Args:
            category_id: Category ID (int) — resolved via get_category_id()
            skip: Number of results to skip
            limit: Maximum number of results

        Returns:
            List of product dictionaries
        """
        url = f"{self.base_url}/product/by-category/{category_id}"
        params = {"skip": skip, "limit": limit}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    products = response.json()
                    return [{"product_id": p["product_id"], "score": 1.0, **p} for p in products]
            except Exception as e:
                logger.error(f"Error searching by category: {e}")

        return []

    async def search_by_params(
        self,
        category_id: Optional[int] = None,
        brand: Optional[str] = None,
        name: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> List[Dict]:
        """Search products by various parameters.

        Args:
            category_id: Category ID (int) — use get_category_id() to resolve
            brand: Brand name filter (searches within product name)
            name: Product name filter (partial match)
            skip: Number of results to skip
            limit: Maximum number of results

        Returns:
            List of product dictionaries
        """
        url = f"{self.base_url}/product/search"
        params = {"skip": skip, "limit": limit}

        if category_id is not None:
            params["category_id"] = category_id
        if brand:
            params["brand"] = brand
        if name:
            params["name"] = name

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    products = response.json()
                    return [{"product_id": p["product_id"], "score": 1.0, **p} for p in products]
            except Exception as e:
                logger.error(f"Error searching by params: {e}")

        return []

    # ------------------------------------------------------------------
    # Product getters
    # ------------------------------------------------------------------
    async def get_product(self, product_id: int) -> Optional[Dict]:
        """Get a single product by ID."""
        url = f"{self.base_url}/product/{product_id}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                logger.error(f"Error getting product {product_id}: {e}")

        return None

    async def get_products_by_ids(self, product_ids: List[int]) -> Dict[int, Dict]:
        """Get multiple products by IDs."""
        products = {}
        for product_id in product_ids:
            product = await self.get_product(product_id)
            if product:
                products[product_id] = product
        return products

    async def get_all_products_with_images(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dict]:
        """Get all products with image URLs for indexing."""
        url = f"{self.base_url}/product/"
        params = {"skip": skip, "limit": limit}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    products = response.json()
                    return [p for p in products if p.get("image_url")]
            except Exception as e:
                logger.error(f"Error getting products: {e}")

        return []

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        """Check if main API is healthy."""
        url = f"{self.base_url}/health"

        async with httpx.AsyncClient(timeout=2.0) as client:
            try:
                response = await client.get(url)
                return response.status_code == 200
            except Exception:
                return False