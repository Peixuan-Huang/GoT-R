from getLLMAnswer import getLLMAnswer
from prompts import GEN_ANSWER_BY_REASONING_PATH_PTOMPT_1003,GEN_ANSWER_BY_REASONING_PATH_PTOMPT_MED_CHINESE,GEN_ANSWER_BY_REASONING_PATH_PTOMPT_MED

# step5 推理路径--LLM-->final answer
def getFinalAnswer(model,question,reasoning_paths):
    temps = []
    for t in reasoning_paths:
      # print(t)
      try:
        temp = ", ".join(t)
      except Exception as e:
        # print(e)
        continue
      else:
        # print(temp)
        temps.append(temp)
    reasoning_paths_str = "\n".join(temps)

    prompt = GEN_ANSWER_BY_REASONING_PATH_PTOMPT_1003 % (question,reasoning_paths_str)
    # print(prompt)
    # return ""
    return getLLMAnswer(model,prompt)
