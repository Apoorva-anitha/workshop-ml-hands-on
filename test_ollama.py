import httpx
import asyncio
import json

async def test_generate():
    url = "http://localhost:8000/generate"
    payload = {
        "prompt": "Why is the sky blue?"
    }
    
    print(f"Sending request to {url}...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("Response from FastAPI:")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(test_generate())
