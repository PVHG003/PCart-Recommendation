### GET /

```shell
curl http://localhost:8000
```

Response:

```json
{
  "message": "Recommendation API is running",
  "version": "1.0",
  "status": "healthy",
  "models": {
    "content_based": true,
    "collaborative": true,
    "hybrid": true
  }
}
```

### GET /api/recommendations/similar/{product_id}

Description: Get products similar to a specific product (Content-Based Filtering)
Use Case: "Customers who viewed this also viewed..."

Parameters:

- product_id (path) - Product ID (required)
- limit (query) - Number of recommendations (default: 10)

```shell
# Get 5 similar products
curl "http://localhost:8000/api/recommendations/similar/prod_123?limit=5"
```

Response:

```json
{
  "recommendations": [
    {
      "id": "prod_456",
      "name": "Blue Running Shoes",
      "category": "shoes",
      "price": 79.99,
      "mrp": 99.99,
      "images": [
        "image1.jpg",
        "image2.jpg"
      ],
      "storeId": "store_1",
      "score": 0.85
    },
    {
      "id": "prod_789",
      "name": "Sports Sneakers",
      "category": "shoes",
      "price": 69.99,
      "mrp": 89.99,
      "images": [
        "image3.jpg"
      ],
      "storeId": "store_2",
      "score": 0.78
    }
  ],
  "algorithm": "content-based",
  "count": 2
}
```

### GET /api/recommendations/personalized/{user_id}

Description: Get personalized recommendations for a user (Collaborative Filtering)
Use Case: "Recommended for you" / "You might also like"

Parameters:

- user_id (path) - User ID (required)
- limit (query) - Number of recommendations (default: 10)

```shell
curl "http://localhost:8000/api/recommendations/personalized/user_abc123?limit=10"
```

```json
{
  "recommendations": [
    {
      "id": "prod_999",
      "name": "Wireless Headphones",
      "category": "electronics",
      "price": 149.99,
      "mrp": 199.99,
      "images": [
        "headphone1.jpg"
      ],
      "storeId": "store_5",
      "score": 4.52
    }
  ],
  "algorithm": "collaborative",
  "count": 10
}
```

### GET /api/recommendations/hybrid

Description: Get hybrid recommendations (combines both collaborative + content-based)
Use Case: Best overall recommendations combining user behavior and product similarity

Parameters:

- user_id (query) - User ID (optional)
- product_id (query) - Product ID (optional)
- limit (query) - Number of recommendations (default: 10)

```shell
# User + Product
curl "http://localhost:8000/api/recommendations/hybrid?user_id=user_123&product_id=prod_456&limit=8"
```

```shell
# User only
curl "http://localhost:8000/api/recommendations/hybrid?user_id=user_123&limit=10"
```

```shell
# Product only
curl "http://localhost:8000/api/recommendations/hybrid?product_id=prod_456&limit=8"
```

```shell
# Neither (default to popular)
curl "http://localhost:8000/api/recommendations/hybrid?limit=10"
```

Response:
```json
{
  "recommendations": [
    {
      "id": "prod_888",
      "name": "Smart Watch",
      "category": "electronics",
      "price": 299.99,
      "mrp": 399.99,
      "images": ["watch1.jpg", "watch2.jpg"],
      "storeId": "store_3",
      "score": 0.92
    }
  ],
  "algorithm": "hybrid",
  "count": 8
}
```

### GET /api/recommendations/popular
Description: Get popular/trending products
Use Case: Homepage trending section, new arrivals
Parameters:
- limit (query) - Number of products (default: 10)

```shell
curl "http://localhost:8000/api/recommendations/popular?limit=20"
```

Response:
```json
{
  "recommendations": [
    {
      "id": "prod_111",
      "name": "Latest iPhone",
      "category": "electronics",
      "price": 999.99,
      "mrp": 1099.99,
      "images": ["iphone1.jpg"],
      "storeId": "store_1",
      "inStock": true
    }
  ],
  "algorithm": "popular",
  "count": 20
}
```