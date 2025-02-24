import re
from neo4j import GraphDatabase
from CypherUtils import queryNode
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

#     top_n=200,
reranker = FlagEmbeddingReranker(
    top_n = 200,
    model="reranker/bge-reranker-large",
    use_fp16=False
)

def getEmbedding(model,text,device):
    with torch.no_grad():
        # print(text)
        embedding = model.encode(text,convert_to_tensor=True,device=device)
    return embedding.cpu().numpy()

def parseKeyEntities(answer):
#   print(answer)
  keyEntities = re.search(r'{key entities}: \[(.*)\]',answer,re.DOTALL).group(1).strip("'").split("', '")
  return keyEntities

def mapEntity2KG(driver,entities,model,device):
    result = []
    with driver.session() as session:
        for entity in entities:
            entity.replace('\'','')
            entity_emb = getEmbedding(model,entity,device)
            # print(entity_emb)
            k = 1 # 匹配的实体类别个数
            matched_node = session.execute_read(queryNode,entity,entity_emb,k)
            result.extend(matched_node)
    return result

def findShortestReasoningPath(tx,start_entity,end_entity,k):
    query1 = (
        f"MATCH (start_entity:Node), (end_entity:Node) "
        f"WHERE elementId(start_entity) = '{start_entity['eid']}' AND elementId(end_entity) = '{end_entity['eid']}' "
        f"MATCH p = allShortestPaths((start_entity)-[*..{k}]->(end_entity)) "
        "RETURN p"
    )
    query2 = (
        f"MATCH p = (start_entity:Node)-[*..{k}]->(end_entity:Node) "
        f"WHERE elementId(start_entity) = '{start_entity['eid']}' AND elementId(end_entity) = '{end_entity['eid']}' "
        f"RETURN p "
    )
    if start_entity['mid'] != end_entity['mid']:
        result = list(tx.run(query1))
    else:
        result = list(tx.run(query2))
    return result

def findNeighborBranchOut(tx,start_entity,k):
    result = []
    if k==1:
        query = (
            f"MATCH (n:Node)-[r]->(m) "
            f"WHERE elementId(n) = '{start_entity['eid']}' "
            f"RETURN n AS node1, type(r) AS relationship, m AS node2"
        )
        result = list(tx.run(query))
        # result = [((res['node1']['mid'],res['node1']['name']),res['relationship'],(res['node2']['mid'],res['node2']['name'])) for res in result]
        result = [(res['node1']['name'],res['relationship'],res['node2']['name']) for res in result]
    return result

def findNeighborBranchIn(tx,start_entity,k):
    query = (
        f"MATCH (n)-[r]->(m:Node) "
        f"WHERE elementId(m) = '{start_entity['eid']}' "
        f"RETURN n AS node1, type(r) AS relationship, m AS node2"
    )
    result = list(tx.run(query))
    # result = [((res['node1']['mid'],res['node1']['name']),res['relationship'],(res['node2']['mid'],res['node2']['name'])) for res in result]
    result = [(res['node1']['name'],res['relationship'],res['node2']['name']) for res in result]
    return result


# 将路径转化为路径
def parse_path_to_triples(result):
    triples = []
    reasoning_paths = []
    for record in result:
        path = record["p"]
        start_node = path.nodes[0]['name']
        reasoning_path = (start_node,)
        for i in range(len(path.relationships)):
            start_node_for_triple = path.nodes[i]
            end_node = path.nodes[i + 1]
            relationship = path.relationships[i]
            # triples.append(((start_node['mid'],start_node['name']), relationship.type, (end_node['mid'],end_node['name'])))
            reasoning_path += (relationship.type, end_node['name'])
            triples.append((start_node_for_triple['name'], relationship.type, end_node['name']))
        reasoning_paths.append(reasoning_path)
    # print(reasoning_paths)
    return reasoning_paths, triples

def getReasoningPathsByKeyEntities(keyEntities,k1,k2,model,device,graph_link,auth):
    driver = GraphDatabase.driver(graph_link,auth=auth)
    if keyEntities != ['']:
        keyEntitiesInKG = mapEntity2KG(driver,keyEntities,model,device) # node list
    else:
        keyEntitiesInKG = []
    # print(keyEntitiesInKG) #[<Record mid='_m_07nznf' name='Brian Singer' score=1.0>, <Record mid='_m_014lc_' name='Donatra' score=1.0>]
    
    reasoning_paths = []
    neighbor_branches = []
    with driver.session() as session:
        # Get Main Reasoning Path
        if len(keyEntitiesInKG) >= 2:
            # k1 = 3
            CandidateEntitySet = keyEntitiesInKG.copy()
            start_entity = CandidateEntitySet[0]
            CandidateEntitySet.remove(start_entity)
            while CandidateEntitySet:
                for end_entity in CandidateEntitySet:
                    # print(f"{start_entity['name']} {end_entity['name']}")
                    paths = session.execute_read(findShortestReasoningPath,start_entity,end_entity,k1)
                    reasoning_paths.extend(paths)
                start_entity = CandidateEntitySet[0]
                CandidateEntitySet.remove(start_entity)
            # print(reasoning_paths) 
            reasoning_paths,reasoning_triples = parse_path_to_triples(reasoning_paths)
            # reasoning_paths = set(reasoning_paths) # 这一步考虑PageRank剪枝；是转换成triple还是推理路径的形式？
            reasoning_triples = set(reasoning_triples)
        else:
            reasoning_paths = []
            reasoning_triples = set()

        # print("main path down")
        
        # Get Neighbor Node
        # k2 = 1
        for keyEntity in keyEntitiesInKG:
            neighbor_branch_in = session.execute_read(findNeighborBranchOut,keyEntity,k2)
            neighbor_branches.extend(neighbor_branch_in)
            neighbor_branch_out = session.execute_read(findNeighborBranchIn,keyEntity,k2)
            neighbor_branches.extend(neighbor_branch_out)
        neighbor_branches = set(neighbor_branches) # 考虑用LLM剪枝，必须要剪枝
        # print(neighbor_branches)
        # print("neighbor down")
        # print(neighbor_branches)
        # neighbor_branches = set()
    # print(reasoning_paths)
    # print(neighbor_branches)
    
    finalTriples = list(reasoning_triples | neighbor_branches)
    # finalTriples = concatenate_triples(finalTriples)
    # print("元组拼接结果")
    # print(finalTriples)
    
    finalTriples = list(set(reasoning_paths+finalTriples))
    return finalTriples


def pruneTriples(question,finalTriples):
    # 使用reranker进行排序剪枝
    nodes = [NodeWithScore(node=TextNode(text=str(doc))) for doc in finalTriples]
    query = f"{{Question}}: {question}"
    # print("重排query")
    # print(query)
    query_bundle = QueryBundle(query_str=query)
    ranked_nodes = reranker._postprocess_nodes(nodes, query_bundle)
    prunedTriples = []
    # count = 0
    for node in ranked_nodes:
        # count+=1
        if node.score > 0.0:
            # print(node.node.get_content(), "-> Score:", node.score)
            prunedTriples.append(eval(node.node.get_content()))

    return prunedTriples
