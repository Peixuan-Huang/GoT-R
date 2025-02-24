import re
from neo4j import GraphDatabase
from CypherUtils import queryNode,queryRelationshipAndTail
import torch
from getReasoningPaths import getReasoningPathsByKeyEntities,pruneTriples

def remove_duplicates_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        item = (item['node1Name'],item['relationship'],item['node2Name'])        
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def remove_duplicates_preserve_order_for_relation(lst):
    seen = set()
    result = []
    for item in lst:
        item = tuple(item[:-1])      
        if item not in seen:
            if len(item)<=4:
                seen.add(item)
                result.append(item)
            elif len(item)>4 and item[0]!=item[-2]: # 防止多跳检索下检索到错误的关系
                seen.add(item)
                result.append(item)
    return result

def remove_duplicates_preserve_order_for_allknowledge(lst):
    seen = set()
    result = []
    for item in lst: 
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def getEmbedding(model,text,device):
    with torch.no_grad():
        embedding = model.encode(text,convert_to_tensor=True,device=device)
    return embedding.cpu().numpy()

def parseCypher2triple(content):
    completedKG = re.search(r'<step 3> {Completed Knowledge Graph}(.*)<step 4> {Relation Paths}',content,re.DOTALL).group(1)
    # print(completedKG)

    # Dictionary to store node identifiers and their respective labels
    node_dict = {}

    # List to store the triples (subject, relationship, object)
    triples = []

    # Regex patterns
    node_pattern = re.compile(r'CREATE \(([-\w]+):\w+ \{\w+: (\'.*?\'|".*?"|\d+)(.*)\}\)')
    # rel_pattern = re.compile(r'CREATE \(([-\w]+)\)-\[:([-\w]+)( \{.*type: "(.*?)".*\})?\]->\((\w+(:\w+ \{\w+: "(.*?)"(.*)\})?)\)')
    rel_pattern = re.compile(r'CREATE (.*)-\[:([-\w]+)( \{.*\})?\]->(.*)')
    set_pattern = re.compile(r'SET ([-\w]+)\.([-\w]+) = "(.*)"')
    # Extract nodes
    for match in re.finditer(node_pattern, completedKG):
        # print(match.groups())
        identifier, rel,other = match.groups()
        rel = rel.strip('"')
        node_dict[identifier] = rel.replace('\'','')
        if other: # 节点属性转关系
            pattern = r'([-\w]+): (\'.*?\'|".*?"|\d+)'
            # Find all matches using re.findall
            rts = re.findall(pattern, other)
            # print(rts)
            # print([(node_dict[identifier],r,t) for r,t in rts])
            triples.extend([(node_dict[identifier],r,t.strip('"').replace('\'','')) for r,t in rts])

    # print("Node Dictionary:")
    # print(node_dict)

    # Extract relationships
    for match in re.finditer(rel_pattern, completedKG):
        # print(match.groups())
        subject, relationship, tempObj, obj = match.groups()
        if '{' in subject:
            real_node_pattern = re.compile(r'\(([-\w]+):\w+ \{\w+: (\'.*?\'|".*?"|\d+)(.*)\}\)')
            #  print("node")
            #  print(re.search(real_node_pattern,subject).groups())
            hnodeID, hnodeName, hnodeRels = re.search(real_node_pattern,subject).groups()
            hnodeName = hnodeName.strip('"')
            node_dict[hnodeID] = hnodeName
            if hnodeRels:
                pattern = r'([-\w]+): (\'.*?\'|".*?"|\d+)'
                # Find all matches using re.findall
                rts = re.findall(pattern, hnodeRels)
                triples.extend([(hnodeName,r,t.strip('"')) for r,t in rts])
        else:
            pattern = r'\(([-\w]+)[\)|:]'
            #  print(re.search(pattern,subject).groups(1)[0])
            hnodeID = re.search(pattern,subject).groups(1)[0]

        if '{' in obj:
            tnodeID = ''
            # print(obj)
            
            real_node_pattern = re.compile(r'\(([-\w]+):\w+ \{\w+: (\'.*?\'|".*?"|\d+)(.*)\}\)')
            try:
                tnodeID, tnodeName, tnodeRels = re.search(real_node_pattern,obj).groups()
                # print(tnodeID, tnodeName, tnodeRels)
            except Exception as e:
                print("--Parse Triple Error!")
                print(obj)
                print(e)
            else:
                tnodeName = tnodeName.strip('"')
                node_dict[tnodeID] = tnodeName
                if tnodeRels:
                    pattern = r'([-\w]+): (\'.*?\'|".*?"|\d+)'
                    # Find all matches using re.findall
                    rts = re.findall(pattern, tnodeRels)
                    triples.extend([(tnodeName,r,t.strip('"')) for r,t in rts])
        else:
            try:
                pattern = r'\(([-\w]+)[\)|:]'
                #  print(re.search(pattern,obj).groups(1)[0])
                tnodeID = re.search(pattern,obj).groups(1)[0]
                if tnodeID not in node_dict:
                    node_dict[tnodeID] = ''
            except Exception as e:
                print(e)
                # CREATE (eastern)-[:covers]->(["New York", "Washington D.C.", "Atlanta"])
                print(content)
                pattern = r'\[(.*)\]'
                #  print(re.search(pattern,obj).groups(1)[0])
                tnodeID = re.search(pattern,obj).groups(1)[0]
                print(tnodeID)
                for tnodeName in tnodeID.split(", "):
                    triples.append((node_dict[hnodeID],relationship,tnodeName))
        if tnodeID:
            triples.append((node_dict[hnodeID],relationship,node_dict[tnodeID]))
        
        if tempObj:
            pattern = r'\w+: (\'.*?\'|".*?"|\d+)'
            # Find all matches using re.findall
            rts = re.findall(pattern, tempObj)
            # print(rts)
            # print([(node_dict[identifier],r,t) for r,t in rts])
            triples.extend([(node_dict[hnodeID],relationship,t.strip('"').replace('\'','')) for t in rts])
            
    # Extract SET
    for match in re.finditer(set_pattern, completedKG):
        hnodeID, relationship, obj = match.groups()
        triples.append((node_dict[hnodeID],relationship,obj))

        # if type_label:  # if there's a type in the relationship
        #     node_dict[obj] = type_label
        #     triples.append((node_dict[subject], relationship, type_label))
        # else:
        #     triples.append((node_dict[subject], relationship, node_dict[obj]))
    
    # Print the resulting dictionary and list of tuples
    # print("Node Dictionary:")
    # print(node_dict)
    # print("\nRelationships:")
    # print(triples)
    return triples,node_dict

def parseRelationPaths(content,node_dict):
    relation_paths_content = re.search(r'<step 4> {Relation Paths}:(.*)',content,re.DOTALL).group(1)
    # print("relation_paths_content:",relation_paths_content)
    node_pattern = re.compile(r'\(([-\w]+).*?\)')
    # rel_pattern = re.compile(r'CREATE \(([-\w]+)\)-\[:([-\w]+)( \{.*type: "(.*?)".*\})?\]->\((\w+(:\w+ \{\w+: "(.*?)"(.*)\})?)\)')
    rel_pattern = re.compile(r'\[:([-\w]+)( \{.*\})?\]')
    path_pattern = re.compile(r'path\d+: \[(.*)\]?')

    relation_chains = []
    # 正向
    for match_path in re.finditer(path_pattern,relation_paths_content):
        # print(match_path)
        # print("matched_path1:",match_path.groups()[0])
        path = match_path.groups()[0].split('), (')
        # print("matched_path:",path)
        for chain in path:
            # print("Chain 1:",chain)
            if chain[0]=='(' and chain[-1]!=')':
                chain+=')'
            elif chain[0]!='(' and chain[-1]==')':
                chain = '('+chain
            elif chain[0]!='(' and chain[-1]!=')':
                chain = '('+chain+')'
            # print("Chain 2:",chain)
            subject = node_dict[re.search(node_pattern,chain).groups()[0]]
            # print(subject)
            relation_chain = (subject,)
            for match in re.finditer(rel_pattern,chain):
                rel,_ = match.groups()
                # print(rel)
                relation_chain+=(rel,)
            relation_chains.append(relation_chain)
            # print(relation_chain)
    # 逆向
    for match_path in re.finditer(path_pattern,relation_paths_content):
        # print("matched_path1:",match_path.groups()[0])
        path = match_path.groups()[0].strip().split('), (')
        # print("matched_path:",path)
        for chain in path:
            if chain[0]=='(' and chain[-1]!=')':
                chain+=')'
            elif chain[0]!='(' and chain[-1]==')':
                chain = '('+chain
            elif chain[0]!='(' and chain[-1]!=')':
                chain = '('+chain+')'
            # print(chain)
            # print(re.findall(node_pattern,chain))
            subjectID = re.findall(node_pattern,chain)[-1]
            if subjectID not in  node_dict:
                break
            subject = node_dict[subjectID]
            # print(subject)
            relation_chain = (subject,)
            for rel in re.findall(rel_pattern,chain)[::-1]:
                # rel,_ = match.groups()
                rel,_ = rel
                relation_chain+=(rel,)
            relation_chains.append(relation_chain)
            # print(relation_chain)

    return list(set(relation_chains))
        


def triples2RelationPaths(triples):
    # triples = [('Brian Singer', 'ACTED_IN', 'Movie 1'), ('Movie 1', 'MEDIUM', 'Film'), ('Brian Singer', 'ACTED_IN', 'Movie 2'), ('Movie 2', 'MEDIUM', 'TV Series'), ('Brian Singer', 'DIRECTED', 'Movie 1'), ('Brian Singer', 'PRODUCED', 'Movie 2')]
    # triples =  [('Justin Bieber', 'SIBLING_OF', 'Jaxon Bieber'), ('Justin Bieber', 'PARENT', 'Jeremy Bieber'), ('Jeremy Bieber', 'CHILDREN', 'Justin Bieber'), ('Jaxon Bieber', 'PARENT', 'Jeremy Bieber'), ('Jeremy Bieber', 'CHILDREN', 'Jaxon Bieber'), ('Justin Bieber', 'PARENT', 'Pattie Mallette'), ('Pattie Mallette', 'CHILDREN', 'Justin Bieber'), ('Jaxon Bieber', 'PARENT', 'Pattie Mallette'), ('Pattie Mallette', 'CHILDREN', 'Jaxon Bieber')]
    # Step 1: Create a dictionary to map objects to their types
    obj_to_type = {}
    for subject, relationship, obj in triples:
        obj_to_type[subject] = {}
        obj_to_type[subject]["relation"] = relationship
        obj_to_type[subject]["obj"] = obj
    # print(obj_to_type)
 
    # Step 2: Create the reasoning chains by replacing objects with their types if available
    relation_chains = []
    for subject, relationship, obj in triples:
        relation_chain = (subject,relationship)
        visited = set()  # Initialize a set to keep track of visited objects
        while obj in obj_to_type:
            if obj in visited:
               break
            visited.add(obj)
            relation_chain += (obj_to_type[obj]["relation"],)
            obj = obj_to_type[obj]["obj"]
        relation_chains.append(relation_chain)
    # print(list(set(relation_chains)))
    return list(set(relation_chains))


# step2: 生成伪图，获取关系（粗粒度和细粒度关系、补充实体）--KG-->knowledge
def getKnowledgeByRelationAndEntity(question,content,model,device,graph_link,auth):
    candidate_entity = set()
    # Parse Cypher to Triples
    triples,node_dict = parseCypher2triple(content)
    # print("-------------------")
    # print("1 解析三元组".center(50,'='))
    # print("triples: ",triples)
    
    # RUR
    for triple in triples:
        candidate_entity.add(triple[0])
        candidate_entity.add(triple[2])
    
    k1 = 3 # l
    k2 = 1 # neighbor
    reasoning_paths_for_original_entity = getReasoningPathsByKeyEntities(candidate_entity,k1,k2,model,device,"bolt://localhost:7687",('neo4j','XXXXXXXX'))
    # for record in reasoning_paths_for_original_entity:
    #     print(record)
    topK = 100
    reasoning_paths_for_original_entity = pruneTriples(question,reasoning_paths_for_original_entity)[:topK]
    
    # RAR
    reasoning_paths_for_original_entity = [] # 记得删掉
    
    # Connect Neo4j
    driver = GraphDatabase.driver(graph_link,auth=auth)


    allKnowledge = []
    relationship_threshold = 0
    with driver.session() as session:
        # step1 retrieved by relation paths
        k1 = 3 # 匹配的实体个数 beamsize
        k2 = 3 # 匹配的关系个数 
        alpha = 0.0 
        # relation_paths = triples2RelationPaths(triples)
        relation_paths = parseRelationPaths(content,node_dict)
        
        # print("2 解析关系路径".center(50,'='))
        # print(relation_paths)
        # print("relation_paths:",relation_paths)
        # 按关系路径进行检索
        reasoning_paths_for_relation = []
        # 正向检索
        for relation_path in relation_paths:
            subject = relation_path[0]
            # candidate_entity.add(subject)
            if not subject:
                break
            #  subject = 'Keyshia Cole'
            subject_emb = getEmbedding(model,subject,device)
            # if subject == "Father Chris Riley":
            #     print(list(subject_emb))
            matched_subject_nodes = session.execute_read(queryNode,subject,subject_emb,k1)
            # print(matched_subject_nodes)
            # print(len(matched_subject_nodes))
            reasoning_path = {}
            for msn in matched_subject_nodes:
                reasoning_path[msn['eid']] = [[msn['name']]]
            # print("每条路径的开头:",reasoning_path)
            for i in range(1,len(relation_path)):
                relation = relation_path[i]
                # reasoning_path_by_relation.append(relation)
                relation_emb = getEmbedding(model,relation,device)
                matched_subject_nodes_temp = []
                reasoning_path_new = {}
                for subject_node in matched_subject_nodes:
                    start_node_eid = subject_node['eid']
                    # print("头实体:",subject_node['name'])
                    # print(subject_node)
                    # print(subject_node['mid'])
                    relationsAndTails = session.execute_read(queryRelationshipAndTail,"Node",start_node_eid,relation_emb,k2,relationship_threshold)
                    # print('relationAndtails:',relationsAndTails)
                    # 剪枝方法1：按路径累积得分
                    for record in relationsAndTails:
                        # print((record['node1Name'],record['relationship'],record['node2Name']))
                        # input()
                        # print(subject_node['score'])
                        # print(record['similarity'])
                        reasoning_path_new[record['node2Eid']] = reasoning_path_new.get(record['node2Eid'],[])
                        # print(reasoning_path)
                        for pre_path in reasoning_path[subject_node['eid']]:
                            if i == 1:
                                reasoning_path_new[record['node2Eid']].append(pre_path+[record['relationship'],record['node2Name'],(record['similarity'] + alpha*subject_node['score']*i)/i]) # (record['similarity'] + alpha*subject_node['score']*i)/(i+1)
                            else:
                                reasoning_path_new[record['node2Eid']].append(pre_path[:-1]+[record['relationship'],record['node2Name'],(record['similarity'] + subject_node['score']*i)/(i+1)])
                        # print("添加后的路径:",reasoning_path_new)
                        if i == 1:
                            record['similarity'] = (record['similarity'] + alpha*subject_node['score']*i)/i # (record['similarity'] + alpha*subject_node['score']*i)/(i+1)
                        else:
                            record['similarity'] = (record['similarity'] + subject_node['score']*i)/(i+1)
                    
                    
                    allKnowledge.extend(relationsAndTails)
                    matched_subject_nodes_temp.extend([{'eid':record["node2Eid"],'name':record["node2Name"],'score':record['similarity']} for record in relationsAndTails])
                matched_subject_nodes = matched_subject_nodes_temp
                # print(len(matched_subject_nodes))
                if len(matched_subject_nodes) > 500:
                    break
                reasoning_path = reasoning_path_new
                # print(reasoning_path)
            reasoning_paths_for_relation.extend([it for item in reasoning_path.values() for it in item])

        topK = 100
        # print("reasoning_paths_for_relation1\n", reasoning_paths_for_relation)
        reasoning_paths_for_relation = [path for path in reasoning_paths_for_relation if type(path[-1]) == float]
        # print("reasoning_paths_for_relation2\n", reasoning_paths_for_relation)
        reasoning_paths_for_relation = sorted(reasoning_paths_for_relation,key=lambda x:x[-1],reverse=True)
        # print("reasoning_paths_for_relation3\n", reasoning_paths_for_relation)
        reasoning_paths_for_relation = remove_duplicates_preserve_order_for_relation(reasoning_paths_for_relation)[:topK]


        # step2 fix graph
        
        for h,r,t in triples:
            # print("正向")
            # print(h,r,t)
            # (h,r,x)
            h_emb = getEmbedding(model,h,device)
            k1 = 3 # 匹配的实体个数
            k2 = 3 # 匹配的关系个数
            matched_head_nodes = session.execute_read(queryNode,h,h_emb,k1)
            # print("matched_head_nodes:")
            # print(matched_head_nodes) # [<Record mid='_m_07nznf' name='Brian Singer' score=1.0>]

            r_emb = getEmbedding(model,r,device)
            for head_node in matched_head_nodes:
                relationsAndTails = session.execute_read(queryRelationshipAndTail,"Node",head_node['eid'],r_emb,k2,relationship_threshold)
                # print("relationsAndTails:")
                # print(relationsAndTails)
                # for record in relationsAndTails:
                #     print((record['node1Name'],record['relationship'],record['node2Name']))
                '''
                [<Record node1='Brian Singer' relationship='_film_actor_film_film_performance_film' node2='X-Men 2 (movie)' similarity=0.836180178497556>, 
                <Record node1='Brian Singer' relationship='_film_actor_film_film_performance_film' node2='Donatra' similarity=0.836180178497556>]
                '''
                allKnowledge.extend(relationsAndTails)
            
            # print("逆向")
            # (x,r,t)
            t_emb = getEmbedding(model,t,device)
            matched_tail_nodes = session.execute_read(queryNode,t,t_emb,k1)
            # print("matched_tail_nodes:")
            # print(matched_tail_nodes)
                    
            r_emb = getEmbedding(model,r,device)
            for tail_node in matched_tail_nodes:
                relationsAndHeads = session.execute_read(queryRelationshipAndTail,"Node",tail_node['eid'],r_emb,k2,relationship_threshold,mode='right')
                # for record in relationsAndTails:
                #     print((record['node1Name'],record['relationship'],record['node2Name']))
                #   print(relationsAndHeads)
                allKnowledge.extend(relationsAndHeads)

        topK = 100
        allKnowledge = sorted(allKnowledge,key=lambda x:x['similarity'],reverse=True)
        allKnowledge = remove_duplicates_preserve_order(allKnowledge)[:topK]

        
        # 将三个检索的内容拼起来
        allKnowledge = reasoning_paths_for_relation+allKnowledge+reasoning_paths_for_original_entity
        allKnowledge = remove_duplicates_preserve_order_for_allknowledge(allKnowledge)

    return allKnowledge


    