import requests

url = "http://localhost:8000/send-resume-request"
data = {
    "name": "Test User",
    "email": "jamesw@icloud.com",
    "message": "Testing"
}

response = requests.post(url, json=data)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

