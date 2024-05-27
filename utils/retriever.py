import requests


def download_file(url: str, filename: str):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)
