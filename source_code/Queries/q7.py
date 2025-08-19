# q7
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter

exports=pd.read_csv('/content/export_processed.csv',encoding='latin1')
imports=pd.read_csv('/content/import_processed.csv',encoding='latin1')

exports['TradeDependencyIndex']=pd.to_numeric(exports['TradeDependencyIndex'],errors='coerce')
imports['TradeDependencyIndex']=pd.to_numeric(imports['TradeDependencyIndex'],errors='coerce')
exports=exports[exports['refYear']==2010]
imports=imports[imports['refYear']==2010]

top_countries=exports.groupby('reporterISO')['CountryTotalPrimaryValue'].sum().nlargest(25).index
exports=exports[exports['reporterISO'].isin(top_countries)&exports['partnerISO'].isin(top_countries)]
imports=imports[imports['reporterISO'].isin(top_countries)&imports['partnerISO'].isin(top_countries)]
export_edges=exports[['reporterISO','partnerISO','TradeDependencyIndex']].copy()
import_edges=imports[['partnerISO','reporterISO','TradeDependencyIndex']].copy()
import_edges.columns=['reporterISO','partnerISO','TradeDependencyIndex']

edges=pd.concat([export_edges,import_edges],ignore_index=True)
edges=edges.groupby(['reporterISO','partnerISO'])['TradeDependencyIndex'].mean().reset_index()

G=nx.DiGraph()
for _,row in edges.iterrows():
    G.add_edge(row['reporterISO'],row['partnerISO'],weight=row['TradeDependencyIndex'])

G.add_nodes_from(top_countries)

degree_centrality=nx.degree_centrality(G)
in_degree_centrality=nx.in_degree_centrality(G)
out_degree_centrality=nx.out_degree_centrality(G)
betweenness_centrality=nx.betweenness_centrality(G,weight='weight')
pagerank=nx.pagerank(G,weight='weight')

most_central_country=max(pagerank.items(),key=itemgetter(1))[0]
print(f"Most central country (by PageRank): {most_central_country}")
print("\nTop 5 countries by PageRank:")
for country,score in sorted(pagerank.items(),key=itemgetter(1),reverse=True)[:5]:
    print(f"{country}: {score:.4f}")
print("\nTop 5 countries by Betweenness Centrality:")
for country,score in sorted(betweenness_centrality.items(),key=itemgetter(1),reverse=True)[:5]:
    print(f"{country}: {score:.4f}")
print("\nTop 5 countries by In-Degree Centrality:")
for country,score in sorted(in_degree_centrality.items(),key=itemgetter(1),reverse=True)[:5]:
    print(f"{country}: {score:.4f}")
print("\nTop 5 countries by Out-Degree Centrality:")
for country,score in sorted(out_degree_centrality.items(),key=itemgetter(1),reverse=True)[:5]:
    print(f"{country}: {score:.4f}")

plt.figure(figsize=(12,8))
pos=nx.spring_layout(G)
nx.draw(G,pos,with_labels=True,node_size=500,node_color='lightblue',font_size=10,font_weight='bold',edge_color='gray')
edge_labels=nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels={(u,v):f"{d['weight']:.3f}" for u,v,d in G.edges(data=True)})
plt.title("Global Trade Network (Top 25 Countries)")
plt.show()
G_disrupted=G.copy()
G_disrupted.remove_node(most_central_country)
print(f"\nSimulating disruption by removing {most_central_country}")

pagerank_disrupted=nx.pagerank(G_disrupted,weight='weight')
betweenness_disrupted=nx.betweenness_centrality(G_disrupted,weight='weight')
in_degree_disrupted=nx.in_degree_centrality(G_disrupted)
out_degree_disrupted=nx.out_degree_centrality(G_disrupted)

print(f"\nNetwork properties before disruption:")
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
print(f"\nNetwork properties after removing {most_central_country}:")
print(f"Nodes: {G_disrupted.number_of_nodes()}, Edges: {G_disrupted.number_of_edges()}")

print("\nTop 5 countries by PageRank after disruption:")
for country,score in sorted(pagerank_disrupted.items(),key=itemgetter(1),reverse=True)[:5]:
    print(f"{country}: {score:.4f}")

plt.figure(figsize=(12,8))
pos_disrupted=nx.spring_layout(G_disrupted)
nx.draw(G_disrupted,pos_disrupted,with_labels=True,node_size=500,node_color='lightcoral',font_size=10,font_weight='bold',edge_color='gray')
edge_labels_disrupted=nx.get_edge_attributes(G_disrupted,'weight')
nx.draw_networkx_edge_labels(G_disrupted,pos_disrupted,edge_labels={(u,v):f"{d['weight']:.3f}" for u,v,d in G_disrupted.edges(data=True)})
plt.title(f"Global Trade Network After Removing {most_central_country}")
plt.show()
