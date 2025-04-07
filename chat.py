import requests
import json

class ChatBot:
    def __init__(self,start_prompt,model):
        self.start_prompt = start_prompt
        self.model = model

    def chat_with_ollama(self,prompt):
        url = "http://localhost:11434/api/generate"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "prompt": prompt
        }

        try:
            response = requests.post(url, data=json.dumps(data), headers=headers)
            response.raise_for_status()  # 检查HTTP错误
            char_list = []
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        char_list.append(json.loads(decoded_line)['response'].strip())
                    except:
                        continue
            return ''.join(char_list)
        except requests.exceptions.RequestException as e:
            print(f"Error occurred: {e}")
            return None

    def run(self,data):
        res=[]
        for i in data:
            prompt = self.start_prompt + i
            res.append(self.chat_with_ollama(prompt))
        return res