import time
import requests

while True:
	time.sleep(60)
	try:
		print(requests.get("http://8.222.130.100").content)
	except:
		print("Server not up")