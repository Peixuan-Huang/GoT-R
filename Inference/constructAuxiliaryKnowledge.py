from getLLMAnswer import getLLMAnswer
from prompts import CONSTRUCT_AUXILIARY_INFORMATION_PROMPT,CONSTRUCT_AUXILIARY_INFORMATION_PROMPT_CWQ,CONSTRUCT_AUXILIARY_INFORMATION_CHINESE_PROMPT

def getAuxiliaryKnowledge(model,question,selected_atomic_relations):
    prompt = CONSTRUCT_AUXILIARY_INFORMATION_PROMPT % (question,selected_atomic_relations)
    return getLLMAnswer(model,prompt)
