import requests
import json

# Set the API endpoint URLs
# detect_spam_url = "https://ig-scam-checker-obfcb.ondigitalocean.app/scam/"
detect_spam_url = "http://localhost:5000/scam/"


# Set the input text message
message = "While transhumanism may offer us a way to escape the limitations of our biology, it may also lead to a loss of the very qualities that make us alive and human. Let's embrace our biology and use it to create a better world. #SayNOTOTranshumanism #StayHuman #StayHuman #CiprianPater #NWO #awakening"

# Define the data to be sent in the POST request
data = {"comment_id": 123, "comment_text": message}

print(message)

# Send the POST request to the detect-spam endpoint
response = requests.post(detect_spam_url, data=json.dumps(data), headers={"Content-Type": "application/json"})

# Print the response from the server
print(response.json())
