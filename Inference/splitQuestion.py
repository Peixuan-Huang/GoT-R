from prompts import QUESTION_DECOMPOSITION_PROMPT, QUESTION_DECOMPOSITION_PROMPT_CWQ, QUESTION_DECOMPOSITION_CHINESE_NEW_PROMPT
from LLM import ChatGPT
import re

def splitQuestion(question):
    prompt = QUESTION_DECOMPOSITION_PROMPT%(question)
    chatgpt = ChatGPT.ChatGPTAPI()
    response = chatgpt.request(prompt)
    try:
        abstract_atomic_questions = re.findall(r'(\[\'.*\'\])',response)[-1]
    except Exception as e:
        print(e)
        abstract_atomic_questions = f"['{question}']"
    abstract_atomic_questions = abstract_atomic_questions.lstrip("['").rstrip("']").split("', '")
    return abstract_atomic_questions


        