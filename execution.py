import requests

def send_post_request(ml_file: str, data_path: str):
    url = "http://localhost:5050/submit"
    headers = {"Content-Type": "application/json"}
    data = {"ml_file": f"/user/{ml_file}", "data_file_path": f"/user/{data_path}"}

    # POST 요청 보내기
    response = requests.post(url, headers=headers, json=data)

    # error message 확인하기
    if response.status_code != 200:
        print(f"Error occurred: {response.text}")
    else:
        print(response.text)

if __name__ == "__main__":
    ml_file = input("Enter the name of the ML file to be executed: ")
    data_path = input("Enter the path to the data: ")
    send_post_request(ml_file, data_path)