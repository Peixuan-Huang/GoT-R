import os
from openai import OpenAI
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, )

os.environ["OPENAI_API_KEY"]="XXXXXXX"

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"


class ChatGPTAPI:
    def __init__(self,temperature=0):
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model = "gpt-3.5-turbo"
        self.max_token = 100
        self.temparature = temperature
        self.seed = 0
        self.sleeptime = 0.05 
        self.message = []

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def request(self,content,haveMemory=False):
        if haveMemory:
            self.message.append({"role": "user","content":content})
        else:
            self.message = [
                {"role": "user","content":content}
            ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            # max_tokens=self.max_token,
            temperature=self.temparature,
            seed=self.seed,
            messages=self.message
        )
        # time.sleep(self.sleeptime)
        LLMResponse = response.choices[0].message.content
        if haveMemory:
            self.message.append({"role": "assistant", "content": LLMResponse})
        return  LLMResponse
