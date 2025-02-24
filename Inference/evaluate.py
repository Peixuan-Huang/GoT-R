import json
from splitQuestion import splitQuestion
from selectAtomicRelation import selectAtomicRelations
from constructAuxiliaryKnowledge import getAuxiliaryKnowledge
from getKnowledgeByRelationAndEntity import getKnowledgeByRelationAndEntity
from getFinalAnswer import getFinalAnswer
from tqdm import tqdm
import argparse
import re
import os
import traceback
import final_encoder.pcl.builder
from easydict import EasyDict
import torch
from sentence_transformers import SentenceTransformer

# hits@1 F1
def calculate_f1_new(true_answers_list, predicted_answers_list):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    count = 0
    for true_answers, predicted_answers in zip(true_answers_list, predicted_answers_list):
        # precision
        count1 = 0
        for predictA in predicted_answers:
            for trueA in true_answers:
                if trueA and (predictA in trueA or trueA in predictA):
                    count1+=1
        precision = count1/len(predicted_answers) if len(predicted_answers) > 0 else 0
        
        # recall
        recall = count1/len(true_answers) if len(true_answers) > 0 else 0
        
        # f1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        count += 1

    return total_precision / count, total_recall / count, total_f1 / count

def calculate_hits_at_1(true_answers_list, predicted_answers_list):
    correct_top_predictions = 0
    
    for true_answers, predicted_answers in zip(true_answers_list, predicted_answers_list):
        for trueA in true_answers:
            if trueA == "":
                continue
            if predicted_answers and (predicted_answers[0] in trueA or trueA in predicted_answers[0]):
                correct_top_predictions += 1
                break     

    hits_at_1 = correct_top_predictions / len(true_answers_list)
    return hits_at_1

def calculate_hits(true_answers_list, predicted_answers_list):
    count1 = 0 # strictly
    count2 = 0 # not strictly
    count3 = 0
    correct_top_predictions = 0

    true_answer_index_list = []
    false_answer_index_list = [i for i in range(len(true_answers_list))]
    i = 0
    for true_answers, predicted_answers in zip(true_answers_list, predicted_answers_list):
        flag = False
        for trueA in true_answers:
            if predicted_answers:
                for predictA in predicted_answers:
                    if predictA == trueA:
                        count1+=1
                        correct_top_predictions += 1
                        flag = True
                        true_answer_index_list.append(i)
                        false_answer_index_list.remove(i)
                        break
                if not flag:
                    for predictA in predicted_answers:
                        if trueA !="" and (predictA in trueA or trueA in predictA) : # :
                            count2+=1
                            correct_top_predictions += 1
                            flag = True
                            true_answer_index_list.append(i)
                            false_answer_index_list.remove(i)
                            break
                if flag:
                    break
        if not flag:
            count3+=1
            # print("trueAnswers:",true_answers)
            # print("predictAnswers:",predicted_answers)    
        i+=1

    hits = correct_top_predictions / len(true_answers_list)
    print("strictly match:",count1)
    print("not strictly match:",count2)
    print("not match:",count3)
    return hits,true_answer_index_list,false_answer_index_list


# For WebQSP
def evaluate(args):
    # Regular expression to match the description fields
    # label_pattern = re.compile(r'\(description\s+"?([^")]+)"?\)')
    answer_pattern = re.compile(r'{Answer}: \[([^\]]+)\]')
    answer_pattern_sub = re.compile(r'\[([^\]]+)\]')

    true_answers_list = []
    predicted_answers_list = []
    with open(args.resultsavepath,'r') as f:
        datas = json.load(f)
        for data in tqdm(datas):
            try:
                # matches = label_pattern.findall(data["targetValue"])
                # true_answers_list.append([match.strip('"') for match in matches])
                # print(data["Parses"])
                # print(data["Parses"]["Answers"])
                tempList = set()
                for answerParse in data["Parses"]:
                    for answer in answerParse["Answers"]:
                        if answer["AnswerType"] == "Entity":
                            tempList.add(answer["EntityName"].lower().replace(',','').replace('-','').replace('.','').replace('"','').replace('\'',''))
                            # tempList.append(answer["EntityName"].encode('latin-1').decode('unicode-escape').lower())
                        else:
                            tempList.add(answer["AnswerArgument"].lower().replace(',','').replace('-','').replace('.','').replace('"','').replace('\'',''))
                            # tempList.append(answer["AnswerArgument"].encode('latin-1').decode('unicode-escape').lower())
                true_answers_list.append(list(tempList))
                        
                        
                # true_answers_list.append([answer["EntityName"].lower() for answer in data["Parses"][0]["Answers"]])
                matches = answer_pattern.findall(data["step5"])
                # print(matches)
                if matches:
                    predicted_answers_list.append([match.strip('\'').lower().replace(',','').replace('-','').replace('.','').replace('"','').replace('\'','') for match in matches[0].split(", ")])
                    # predicted_answers_list.append([match.strip('\'').lower() for match in matches[0].split(", ")])
                else:
                    sub_matches = answer_pattern_sub.findall(data["step5"])
                    if sub_matches:
                        predicted_answers_list.append([match.strip('\'').lower().replace(',','').replace('-','').replace('.','').replace('"','').replace('\'','') for match in sub_matches[0].split(", ")])
                    else:
                        predicted_answers_list.append([])
            except Exception as e:
                print(e)
                print(matches)

    # print(true_answers_list)
    # print('----------------------------------------')
    # print(predicted_answers_list)
    precision, recall, f1 = calculate_f1_new(true_answers_list, predicted_answers_list)
    hits_at_1 = calculate_hits_at_1(true_answers_list, predicted_answers_list)
    hits,true_answer_index_list,false_answer_index_list = calculate_hits(true_answers_list, predicted_answers_list)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"Hits@1: {hits_at_1}")
    print(f"Hits: {hits}")

    return true_answer_index_list,false_answer_index_list


def getAtomicRelations(atomic_relation_filepath):
    atomic_relations = []
    with open(atomic_relation_filepath,'r') as f:
        line = f.readline()
        while line:
            atomic_relation = ', '.join(line.strip().split(','))
            atomic_relations.append(atomic_relation)
            line = f.readline()
    return list(atomic_relations)


def test(args):
    auth = (args.Neo4jUserName,args.Neo4jPassword)
    
    # 加载模型
    argsModel = {
        "alpha":10,   # 可调
        "print_freq":1,
        "num_epochs":50,
        "lr": 2e-6,# 2e-6,  # 可调
        "temperature": 0.5,#0.2, # 可调
        "num_cluster": [3446],
        "gpu": 0,
        "topk_neg_cluster": 16, # 可调
        "warmup_epoch": 0,
        "low_dim": 768,
        "seed": 2024,
        "pcl_r":16, #queue size; number of negative pairs; needs to be smaller than num_cluster (default: 16384)
        "moco_m":0.999,   # 可调
        "mlp":False,
        "train_batch_size":16,
        "dev_batch_size":5,
        "test_batch_size":5,
        "cos":False,
        "schedule":[120, 160],   # 可调
        "device_ids":[0,1],
        "train_data_path":r"./datasets/dataset_train.jsonl",
        "dev_data_path": r"./datasets/dataset_dev.jsonl",
        # "test_data_path":r"./datasets/dataset_test.jsonl",
        "atomic_relation_path": r"./datasets/atomic_rtypes.txt",
        "savedir": r"./checkpoints/"
    }
    model_path = r"./checkpoints/question_model.pth"

    argsModel = EasyDict(argsModel)
    question_model = final_encoder.pcl.builder.MoCo(argsModel.low_dim, argsModel.pcl_r, argsModel.moco_m, argsModel.temperature, argsModel.device_ids,argsModel.mlp)
    question_model.load_state_dict(torch.load(model_path,map_location="cuda"))

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define Model
    model_name = args.EncoderName
    model = SentenceTransformer(model_name)
    
    # 加载数据
    atomic_relation_filepath=args.atomicrelationpath
    all_atomic_relations = getAtomicRelations(atomic_relation_filepath)
    
    

    resultList = []
    if os.path.exists(args.resultsavepath):
        with open(args.resultsavepath, 'r') as f:
            resultList = json.load(f)
            resultLen =len(resultList)
    else:
        resultList = []
        resultLen = 0

    with open(args.testfilepath, 'r') as f:
        testdataset = json.load(f) #["Questions"]
    
    try:
        for testdata in tqdm(testdataset[resultLen:],desc="testing WebQuestion"):
            result = testdata.copy()
            question = testdata["RawQuestion"]
            # print(question)
            
            # step0 query--LLM-->Atomicquestion
            abstract_atomic_questions = splitQuestion(question)
            result["step1"] = abstract_atomic_questions.copy()
            
            # step1 query--Encoder-->RelevantAtomicRelations
            selected_atomic_relations = selectAtomicRelations(abstract_atomic_questions,all_atomic_relations,question_model,kq=10)
            result["step2"] = selected_atomic_relations.copy()
            
            # step2 query+relations--LLM-->GoT
            auxiliaryInfo = getAuxiliaryKnowledge(args.LLM, question,selected_atomic_relations)
            result["step3"] = auxiliaryInfo
            
            # step3 GoT--KG-->ReasoningEvidence
            reasoningPaths = []
            try:
                reasoningPaths = getKnowledgeByRelationAndEntity(question,auxiliaryInfo,model,device,args.Neo4jLink,auth)
            except Exception as e:
                traceback.print_exc()
                tryTimes = 0
                while tryTimes < 5:
                    reasoningPaths = getKnowledgeByRelationAndEntity(question,auxiliaryInfo,model,device,args.Neo4jLink,auth)
                    if reasoningPaths:
                        break
            result["step4"] = reasoningPaths.copy()

            # step4 query+ReasoningEvidence-->LLM-->Answer
            finalAnswer = getFinalAnswer(args.LLM, question,reasoningPaths)
            result["step5"] = finalAnswer

            # 保存测试结果
            resultList.append(result)
    finally:
        with open(args.resultsavepath, 'w+') as write_f:
            json.dump(resultList, write_f,indent=4)

    



if __name__=="__main__":
    ########################################
    #                1号机器                #
    ########################################
    # 解析命令行参数
    parser = argparse.ArgumentParser()

    parser.add_argument("--testfilepath", type=str, default="./QA_dataset/test.json", help="file path of test dataset")
    parser.add_argument("--resultsavepath", type=str, default="result.json", help="save path of result")
    parser.add_argument("--atomicrelationpath", type=str, default="datasets/atomic_rtypes.txt", help="file path of atomic rtypes")
    parser.add_argument("--Neo4jLink", type=str, default="bolt://localhost:7687", help="The link of Neo4j Graph DB")
    parser.add_argument("--Neo4jUserName", type=str, default="neo4j", help="The username of Neo4j Graph DB")
    parser.add_argument("--Neo4jPassword", type=str, default="XXXXXXXX", help="The password of Neo4j Graph DB")
    parser.add_argument("--EncoderName", type=str,  default='./sentence_bert/all-MiniLM-L6-v2', help="The name of encoder model (embedding)")
    parser.add_argument("--LLM", type=str,  default='chatgpt', help="The large langguage model")
    args = parser.parse_args()
    
    # 执行方法
    test(args)
    evaluate(args)
    