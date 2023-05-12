import requests
import json

# Set the API endpoint URLs
detect_spam_url = "https://ig-scam-checker-obfcb.ondigitalocean.app/scam/"

# Set the input text message
message = "This is a test."

# Define the data to be sent in the POST request
data = {"comment_id": 123, "comment_text": message}

print(message)

# Send the POST request to the detect-spam endpoint
response = requests.post(detect_spam_url, data=json.dumps(data), headers={"Content-Type": "application/json"})

# Print the response from the server
print(response.json())
