from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import RDFS, XSD
import networkx as nx
import matplotlib.pyplot as plt
from py2neo import Graph as Neo4jGraph

# 初始化RDF图
g = Graph()

# 定义命名空间
ns = "finance"

# 定义实体类型
g.add((URIRef(ns + "Company"), RDF.type, RDFS.Class))
g.add((URIRef(ns + "FinancialIndicator"), RDF.type, RDFS.Class))
g.add((URIRef(ns + "Report"), RDF.type, RDFS.Class))
g.add((URIRef(ns + "Industry"), RDF.type, RDFS.Class))

# 定义关系
g.add((URIRef(ns + "publishes"), RDF.type, RDF.Property))
g.add((URIRef(ns + "contains"), RDF.type, RDF.Property))
g.add((URIRef(ns + "belongsTo"), RDF.type, RDF.Property))
g.add((URIRef(ns + "hasValue"), RDF.type, RDF.Property))

# 添加示例数据
ning_de = URIRef(ns + "Company/NingDe")
g.add((ning_de, RDF.type, URIRef(ns + "Company")))
g.add((ning_de, RDFS.label, Literal("宁德时代")))

revenue = URIRef(ns + "FinancialIndicator/Revenue")
g.add((revenue, RDF.type, URIRef(ns + "FinancialIndicator")))
g.add((revenue, RDFS.label, Literal("营业收入")))

report_2022 = URIRef(ns + "Report/NingDe2022")
g.add((report_2022, RDF.type, URIRef(ns + "Report")))
g.add((report_2022, RDFS.label, Literal("宁德时代2022年报")))

g.add((ning_de, URIRef(ns + "publishes"), report_2022))
g.add((report_2022, URIRef(ns + "contains"), revenue))
g.add((revenue, URIRef(ns + "hasValue"), Literal("xx亿元", datatype=XSD.string)))

# 将RDF图转换为NetworkX图进行可视化
nx_graph = nx.Graph()
for s, p, o in g:
    nx_graph.add_edge(s, o, label=p)

# 使用Neo4j存储图数据（示例代码，需要配置Neo4j连接）
# neo4j_graph = Neo4jGraph("bolt://localhost:7687", auth=("neo4j", "password"))
# 
# # 将RDF图数据导入Neo4j
# for s, p, o in g:
#     cypher_query = (
#         f"MERGE (s {{uri: '{s}'}}) "
#         f"MERGE (o {{uri: '{o}'}}) "
#         f"MERGE (s)-[r {{uri: '{p}'}}]->(o)"
#     )
#     neo4j_graph.run(cypher_query)

print("知识图谱构建完成。")