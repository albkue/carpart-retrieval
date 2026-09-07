import requests

# Configuration
API_URL = "http://localhost:8001/api/v1"
TEST_IMAGE = "tests/test_image.jpg" # You can replace this with any image

def test_health():
    print("Checking service health...")
    try:
        response = requests.get(f"{API_URL}/../../health")
        print(f"Status: {response.json()}")
    except Exception as e:
        print(f"Error connecting to server: {e}")

def test_brands():
    print("\nFetching supported brands...")
    response = requests.get(f"{API_URL}/brands")
    print(f"Supported brands: {response.json()['count']}")

def test_categories():
    print("\nFetching supported categories...")
    response = requests.get(f"{API_URL}/categories")
    print(f"Supported categories: {response.json()['categories']}")

if __name__ == "__main__":
    test_health()
    test_brands()
    test_categories()
    
    print("\nNext step: Try a real search by running:")
    print("curl -X POST http://localhost:8001/api/v1/search-by-image -F 'file=@your_image.jpg'")
