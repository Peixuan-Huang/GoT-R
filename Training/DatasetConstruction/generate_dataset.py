# generate dataset for training encoder according to atomic relationships
import json
from neo4j import GraphDatabase
# from getLLMAnswer import getLLMAnswer
from LLM import ChatGPT
from tqdm import tqdm
import re
from prompts import REWRITE_ONE_HOP_TRIPLET_TO_QUESTION_WITHOUT_ANSWER_MULTI_QUESTION_PROMPT,REWRITE_ONE_HOP_TRIPLET_TO_QUESTION_WITHOUT_ANSWER_MULTI_QUESTION_5_CHINESE_PROMPT
from prompts import REWRITE_TWO_HOP_TRIPLET_TO_QUESTION_WITHOUT_ANSWER_MULTI_QUESTION_PROMPT
import torch
import numpy as np
import numpy as np
import random

def getAtomicRelations(atomic_relation_filepath):
    atomic_relations = []
    with open(atomic_relation_filepath,'r') as f:
        line = f.readline()
        while line:
            atomic_relation = tuple(line.strip().split(','))
            atomic_relations.append(atomic_relation)
            line = f.readline()
    return atomic_relations

    
def getAtomicRelationsByStr(atomic_relation_filepath):
    atomic_relations = []
    with open(atomic_relation_filepath,'r') as f:
        line = f.readline()
        while line:
            atomic_relation = tuple(line.strip().split(','))
            atomic_relations.append(atomic_relation)
            line = f.readline()
    return [", ".join(atomic_relation) for atomic_relation in atomic_relations]

def queryTripletByRelation(tx,relation_path,k):
    if len(relation_path)==1:
        # query = f"MATCH (n)-[r:{relation_path[0]}]->(m) RETURN n,r,m LIMIT {k}"
        query = f"MATCH (h)-[r:{relation_path[0]}]->(t) "\
                f"WHERE elementId(h) <> elementId(t) "\
                f"WITH h, r, t "\
                f"ORDER BY rand() "\
                f"WITH collect([h, r, t]) AS triples "\
                f"UNWIND triples AS triple "\
                f"RETURN triple[0] AS h, triple[1] AS r, triple[2] AS t "\
                f"LIMIT {k}"
        result = tx.run(query)
        result = [(record['h']['name'],record['r'].type,record['t']['name']) for record in result]
        if not result:
            query = f"MATCH (h)-[r:{relation_path[0]}]->(t) RETURN h,r,t LIMIT {k}"
            result = tx.run(query)
            result = [(record['h']['name'],record['r'].type,record['t']['name']) for record in result]
    elif len(relation_path)==2:
        # query = f"MATCH (n)-[r1:{relation_path[0]}]->(m1)-[r2:{relation_path[1]}]->(m2) RETURN n,r1,m1,r2,m2 LIMIT {k}"
        query = f"MATCH (h)-[r1:{relation_path[0]}]->(t1)-[r2:{relation_path[1]}]->(t2) "\
                f"WHERE elementId(h) <> elementId(t2) "\
                f"WITH h, r1, t1, r2, t2 "\
                f"ORDER BY rand() "\
                f"WITH collect([h, r1, t1, r2, t2]) AS triples "\
                f"UNWIND triples AS triple "\
                f"RETURN triple[0] AS h, triple[1] AS r1, triple[2] AS t1, triple[3] AS r2, triple[4] AS t2 "\
                f"LIMIT {k}"
        result = tx.run(query)
        result = [(record['h']['name'],record['r1'].type,record['t1']['name'],record['r2'].type,record['t2']['name']) for record in result]
        if not result:
            query = f"MATCH (h)-[r1:{relation_path[0]}]->(t1)-[r2:{relation_path[1]}]->(t2) RETURN h,r1,t1,r2,t2 LIMIT {k}"
            result = tx.run(query)
            result = [(record['h']['name'],record['r1'].type,record['t1']['name'],record['r2'].type,record['t2']['name']) for record in result]
        
    if not result:
        print(relation_path)
    return result

def samaplingTriplet(atomic_relations,k,uri,auth,savepath):
    triplets = []

    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session() as session:
        for atomic_relation in tqdm(atomic_relations,desc='sampling triplets'):
            subtriplets = session.execute_read(queryTripletByRelation,atomic_relation,k)
            triplets.extend(subtriplets)
    with open(savepath,'w') as f:
        json.dump(triplets,f,indent=4)
    return triplets

def trans2questions(triplets,prompts,k2,save_path):
    chatgpt = ChatGPT.ChatGPTAPI()
    result = []
    for id,triplet in enumerate(tqdm(triplets)):
        # response = llama.askSingle(prompt%(str(triplet)))
        if len(triplet) == 3:
            prompt = prompts[0]%(triplet) # webqsp
            atomic_relation = [triplet[1]]
        elif len(triplet) == 5:
            prompt = prompts[1]%(triplet)
            atomic_relation = [triplet[1],triplet[3]]
        response = chatgpt.request(prompt)
        # print(response)
        # pattern = r"<question>:\s*(.*?\?)"

        # # match = re.search(pattern, response)
        # # if match:
        # #     question = match.group(1)  
        # #     print(question)
        # # else:
        # #     print("No match found")
        questionList = response.lstrip("['").rstrip("']").split("', '")
        for idx,question in enumerate(questionList):
            result_dict = {}
            result_dict['question_id'] = id*k2 + idx + 1
            result_dict['question'] = question
            result_dict['abstract_question'] = question.replace(triplet[0],'<topic entity>')
            result_dict['entity_set'] = [triplet[0]]
            result_dict['ground_truth'] = [triplet[-1]]
            result_dict['pos_triplet'] = triplet
            result_dict['atomic_relation'] = atomic_relation
            result.append(result_dict)
        
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4)
         
    return result


def get_sentence_embedding(tokenizer, model, loader, device):
    embeddings = []
    for batch in tqdm(loader,desc='Processing Bataches'):
        # Tokenize the sentence and get the token ids
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
    
        # Get the output from DistilBERT (last hidden state)
        with torch.no_grad():
            batch_embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)


def getNegativeSampleByCL(questions,atomic_relations,save_dir):
    dataset = []
    for idx,question in enumerate(questions):
        question_dict = question.copy()
        question_dict["question_id"] = idx+1
        question_dict["pos_relation_indexs"] = atomic_relations.index(", ".join(question_dict["atomic_relation"]))
        dataset.append(question_dict)
        with open(save_dir+"dataset.jsonl", 'a') as f:
            json.dump(question_dict, f, indent=4)
            f.write('\n') 
    
    # split train and dev
    atomic = {}
    for data in dataset:
        atomic[','.join(data['atomic_relation'])] = atomic.get(','.join(data['atomic_relation']),[])
        atomic[','.join(data['atomic_relation'])].append(data)
    for v in atomic.values():
        dev_dataset = random.sample(v,int(0.2*len(v)))
        for dev in dev_dataset:
            with open(save_dir+"dataset_dev.jsonl", 'a') as f:
                json.dump(dev, f)
                f.write('\n') 
        for data in v:
            if data not in dev_dataset:
                with open(save_dir+"dataset_train.jsonl", 'a') as f:
                    json.dump(data, f)
                    f.write('\n') 

    

if __name__=="__main__":
    datasetsDir = r"./dataset/WebQSP/"
    uri = "bolt://localhost:7687"
    auth=("neo4j", "XXXXXXXX")
    
    k1 = 3
    k2 = 3
    prompts = [REWRITE_ONE_HOP_TRIPLET_TO_QUESTION_WITHOUT_ANSWER_MULTI_QUESTION_PROMPT,REWRITE_TWO_HOP_TRIPLET_TO_QUESTION_WITHOUT_ANSWER_MULTI_QUESTION_PROMPT]
    # prompts = [REWRITE_ONE_HOP_TRIPLET_TO_QUESTION_WITHOUT_ANSWER_MULTI_QUESTION_5_CHINESE_PROMPT,'%s']
    #######################################################################
    
    # Loading Atomic Relations
    atomic_relation_filepath = r"atomic_rtypes.txt"
    atomic_relations = getAtomicRelations(datasetsDir+atomic_relation_filepath)
    print(len(atomic_relations)) 
    
    # Sampling Triples
    triplets_savename = r"triples.json"
    triplets = samaplingTriplet(atomic_relations,k1,uri,auth,datasetsDir+triplets_savename)
    print(len(triplets))
    
    # Sampling Questions
    questions_savename = r"questions.json"
    questions = trans2questions(triplets,prompts,datasetsDir+questions_savename)
    print(len(questions))
    
    # Constructing Dataset
    atomic_relations = getAtomicRelationsByStr(atomic_relation_filepath)
    print(len(atomic_relations)) 
    getNegativeSampleByCL(questions,atomic_relations,datasetsDir)

    

