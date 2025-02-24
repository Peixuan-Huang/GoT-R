# 查询出相似度最高的所有节点或关系
def queryNode(tx,nodeName,node_emb,k):
  # WebQSP:500 CWQ:1000
  query = (
    f"CALL db.index.vector.queryNodes('NodeIndex', 1000, $node_emb) "
    f"YIELD node AS matchNode, score "
    f"WITH matchNode, score "
    f"ORDER BY score DESC "
    f"WITH collect(score) AS scores, collect(matchNode) AS nodes "
    f"WITH nodes, scores, apoc.coll.sort(apoc.coll.toSet(scores))[-{k}..] AS top_k_scores "
    f"WITH [i IN range(0, size(scores)-1) WHERE scores[i] IN top_k_scores | nodes[i]] AS top_k_nodes, "
    f"[i IN range(0, size(scores) - 1) WHERE scores[i] IN top_k_scores | scores[i]] AS final_scores "
    f"UNWIND range(0, size(top_k_nodes) - 1) AS i "
    f"RETURN elementId(top_k_nodes[i]) as eid, top_k_nodes[i].mid AS mid, top_k_nodes[i].name AS name, final_scores[i] AS score "
  )
  
  # result = list(tx.run(query))
  result = list(tx.run(query,node_emb=node_emb))
  # print(result)
  return result


def queryRelationshipAndTail(tx,nodeLabel,nodeEid,relation_emb,k,threshold,mode="left"):
  # 返回topK相似度的所有关系
  if mode=='left':
    query = (
      f"WITH $relation_emb AS query_embedding "
      f"MATCH (n:{nodeLabel})-[r]->(m:{nodeLabel}) "
      f"WHERE elementId(n)='{nodeEid}' "
      f"WITH n, r, m, gds.similarity.cosine(r.embedding, query_embedding) AS similarity "
      f"WHERE similarity > {threshold} "
      f"WITH n, r, m, similarity "
      f"ORDER BY similarity DESC "
      f"WITH collect(similarity) AS sims, collect(r) AS rels, collect(n) AS nodes1, collect(m) AS nodes2 "
      f"WITH sims, rels, nodes1, nodes2, apoc.coll.sort(apoc.coll.toSet(sims))[-{k}..] AS top_k_sims "
      f"WITH [i IN range(0, size(sims) - 1) WHERE sims[i] IN top_k_sims | rels[i]] AS top_k_rels, "
      f"[i IN range(0, size(sims) - 1) WHERE sims[i] IN top_k_sims | nodes1[i]] AS top_k_nodes1, "
      f"[i IN range(0, size(sims) - 1) WHERE sims[i] IN top_k_sims | nodes2[i]] AS top_k_nodes2, "
      f"[i IN range(0, size(sims) - 1) WHERE sims[i] IN top_k_sims | sims[i]] AS final_sims "
      f"UNWIND range(0, size(top_k_rels) - 1) AS i "
      f"RETURN elementId(top_k_nodes1[i]) AS node1Eid, top_k_nodes1[i].mid AS node1Mid, top_k_nodes1[i].name AS node1Name, "
      f"type(top_k_rels[i]) AS relationship, elementId(top_k_nodes2[i]) AS node2Eid, top_k_nodes2[i].mid AS node2Mid, top_k_nodes2[i].name AS node2Name, final_sims[i] AS similarity"
    )
  elif mode=='right':
    query = (
      f"WITH $relation_emb AS query_embedding "
      f"MATCH (n:{nodeLabel})-[r]->(m:{nodeLabel}) "
      f"WHERE elementId(m)='{nodeEid}' "
      f"WITH n, r, m, gds.similarity.cosine(r.embedding, query_embedding) AS similarity "
      f"WHERE similarity > {threshold} "
      f"WITH n, r, m, similarity "
      f"ORDER BY similarity DESC "
      f"WITH collect(similarity) AS sims, collect(r) AS rels, collect(n) AS nodes1, collect(m) AS nodes2 "
      f"WITH sims, rels, nodes1, nodes2, apoc.coll.sort(apoc.coll.toSet(sims))[-{k}..] AS top_k_sims "
      f"WITH [i IN range(0, size(sims) - 1) WHERE sims[i] IN top_k_sims | rels[i]] AS top_k_rels, "
      f"[i IN range(0, size(sims) - 1) WHERE sims[i] IN top_k_sims | nodes1[i]] AS top_k_nodes1, "
      f"[i IN range(0, size(sims) - 1) WHERE sims[i] IN top_k_sims | nodes2[i]] AS top_k_nodes2, "
      f"[i IN range(0, size(sims) - 1) WHERE sims[i] IN top_k_sims | sims[i]] AS final_sims "
      f"UNWIND range(0, size(top_k_rels) - 1) AS i "
      f"RETURN elementId(top_k_nodes1[i]) AS node1Eid, top_k_nodes1[i].mid AS node1Mid, top_k_nodes1[i].name AS node1Name, "
      f"type(top_k_rels[i]) AS relationship, elementId(top_k_nodes2[i]) AS node2Eid, top_k_nodes2[i].mid AS node2Mid, top_k_nodes2[i].name AS node2Name, final_sims[i] AS similarity"
    )    
  
  # f"LIMIT {k} "
  result = list(tx.run(query,relation_emb=relation_emb))
  result2dict = []
  for res in result:
    result2dict.append(dict(res))
  return result2dict

def queryRelationshipAndTailByINdex(tx,nodeLabel,nodeId,relation_emb,k,threshold):
  query = (
    f"CALL db.index.vector.queryRelationships('RelationshipIndex', {k}, $relation_emb) YIELD rel, score "
    f"MATCH (n: {nodeLabel})-[rel]->(m) "
    f"WHERE elementId(n) = '{nodeId}' "
    f"RETURN n.mid AS node1Mid, n.name AS node1Name, rel.rtype AS relationship, m.mid AS node2Mid, m.name AS node2Name, score "
    f"ORDER BY score DESC "
  )
  result = list(tx.run(query,relation_emb=relation_emb))
  return result


def queyRelationByNodeMid(tx,nodeLabel,nodeMid):
  query = f"MATCH (n:{nodeLabel} {{mid: '{nodeMid}'}})-[r]-() RETURN r"
  result = tx.run(query)
  return result