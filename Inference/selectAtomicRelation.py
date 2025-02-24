import torch
import torch.nn as nn

def getSelectedAtomicRelations(atomic_questions,atomic_relations,question_model,k):
    criterion=nn.CosineSimilarity()
    
    selected_atomic_relations = []
    for question in atomic_questions:
        with torch.no_grad():
            question_embedding = question_model([question], is_eval=True)
            atomic_relations_embedding = question_model(atomic_relations, is_eval=True)
        
        cossim = criterion(question_embedding, atomic_relations_embedding).tolist()
        indexed_lst = list(enumerate(cossim))
        sorted_lst = sorted(indexed_lst, key=lambda x: x[1], reverse=True)
        
        sub_atomic_relations = []
        for index, value in sorted_lst[:k]:
            relation = atomic_relations[index]
            # print(relation,value)
            sub_atomic_relations.append(relation)
            
        selected_atomic_relations.extend(sub_atomic_relations)
    return selected_atomic_relations#list(set(selected_atomic_relations))

def selectAtomicRelations(atomic_questions,atomic_relations,question_model,kq=3):
    selected_atomic_relations = getSelectedAtomicRelations(atomic_questions,atomic_relations,question_model,kq)
    return selected_atomic_relations

