import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import operator
import random
from datetime import datetime, timedelta
try:
    import geopandas as gp #might need to install
    from shapely.geometry import Point
except: #If there is an exception in importing geopandas or shapely, we will ignore it and
        #simply not import the package.
    pass
from collections import defaultdict
from scipy.cluster import hierarchy
from scipy.spatial import distance

def getnodegeodata(nodedata):
    nodedata.columns = ['nodeid','lat','long']
    nodedata['geoloc'] = nodedata.apply(lambda x: Point(x.lat,x.long),axis=1)
    nodedata.index = nodedata.nodeid
    nodedataseries = nodedata.geoloc
    nodegeodict = dict(nodedataseries)
    nodegeodict_1 = {}
    nodegeodict_1['geometry'] = list(pd.Series(nodegeodict))
    return nodegeodict_1

def getedgedata(edgedata,nodes):
    edgedata.columns = ['edgeid','startnode','endnode','distance']
    edgedata = edgedata
    edgelist = list(zip(edgedata.startnode,edgedata.endnode))
    nodestofilter = nodes[:1500]
    filterededges = []
    for item in edgelist:
      if item[0] in nodestofilter and item[1] in nodestofilter:
        filterededges.append(item)
    return filterededges
    
def plot_network(g,node_dist, nodecolor='g',nodesize=1200,nodealpha=0.6,edgecolor='k',\
                 edgealpha=0.2,figsize=(9,6),title=None,titlefontsize=20,savefig=False,\
                 filename=None,bipartite=False,bipartite_colors=None,nodelabels=None,
                 edgelabels=None):
    #pos=nx.spring_layout(g,iterations=200)
    pos=nx.spring_layout(g,k=node_dist,iterations=300)
    nodes=g.nodes()
    edges=g.edges()
    plt.figure(figsize=figsize)
    
    nx.draw_networkx_edges(g,pos=pos,edge_color=edgecolor,alpha=edgealpha)
    print(1)
    #nx.draw_networkx_edges(g,pos=pos,edge_color=edgecolor,alpha=edgealpha)
    if bipartite and bipartite_colors!=None:
        bipartite_sets=nx.bipartite.sets(g)
        _nodecolor=[]
        for _set in bipartite_sets:
            _clr=bipartite_colors.pop()
            for node in _set:
                _nodecolor.append(_clr)

        nx.draw_networkx_nodes(g,pos=pos,node_color=_nodecolor,alpha=nodealpha,node_size=nodesize)
    else:
        nx.draw_networkx_nodes(g,pos=pos,node_color=nodecolor,alpha=nodealpha,node_size=nodesize)

    labels={}
    for idx,node in enumerate(g.nodes()):
        labels[node]=str(node)

    if nodelabels!=None:
        nx.draw_networkx_labels(g,pos,labels,font_size=16)
    if edgelabels!=None: #Assumed that it is a dict with edge tuple as the key and label as value.
        nx.draw_networkx_edge_labels(g,pos,edgelabels,font_size=12)
    plt.xticks([])
    plt.yticks([])
    if title!=None:
        plt.title(title,fontsize=titlefontsize)
    if savefig and filename!=None:
        plt.savefig(filename,dpi=300)
        
def getdegree(graph):
    """ 
        Return Indegrees
    """
    node_degrees=[item for item in dict(graph.degree()).items()]
    return node_degrees

def generate_degree_rank_plot(edges_with_weights):
    g=nx.Graph() #Instantiate an Undirected Graph.
    #Add all edges to DiGraph degardless of weight threshold.
    g.add_edges_from(edges_with_weights)
    
    deg=list(sorted(dict(nx.degree(g)).values(),reverse=True)) 
    fig,ax=plt.subplots(1,1,figsize=(8,6))
    ax.loglog(deg,'b-',marker='o')
    ax.set_ylabel('Degree',fontsize=18)
    ax.set_xlabel('Rank',fontsize=18)

def getfilterededges(edgedata):
  edgedata.columns = ['startnode','endnode']
  edges = list(zip(edgedata.startnode,edgedata.endnode))
  nodestofilter = list(range(1000))
  filterededges = []
  for item in edges:
    if item[0] in nodestofilter and item[1] in nodestofilter:
      filterededges.append(item)
  return filterededges


def get_geodataframe_airports(df,airport_ids):
  geo_df_dict={'geometry':list(),'station_ids':list()}

  for airport_id in airport_ids:  #Iterate over all station_ids.
      _df=df[df['AIRPORT_ID']==airport_id]  #Filter rows where Start Station ID equals stn_id .
      if _df.shape[0]>0:
          lat=_df['LATITUDE'].values[0]  #Get the lat value of the particular station.
          lon=_df['LONGITUDE'].values[0] #Get the lon value of the particular station.
          geo_df_dict['geometry'].append(Point(lon,lat))  #Add this as a Shapely.Point value under the 'geometry' key.
          geo_df_dict['station_ids'].append(airport_id)
              
  geo_df_dict['geometry']=list(geo_df_dict['geometry'])
  geo_stations=gp.GeoDataFrame(geo_df_dict)
  geo_stations.drop(geo_stations[geo_stations['geometry']==Point(0,0)].index,inplace=True)
  geo_stations.reset_index(inplace=True)
  geo_stations.to_crs = {'init': 'epsg:4326'}
  return geo_stations

def generate_clustering_coefficient_plot(g):
    sns.set_style('whitegrid')
    #Ignore nodes with clustering coefficients of zero.
    clustering_coefficients=list(filter
        (lambda y: y[1]>0,sorted(
            nx.clustering(g).items(),key=lambda x: x[1],reverse=True)))
    plt.figure(figsize=(7,7))
    plt.plot(list(map(lambda x: x[1],clustering_coefficients)))
    plt.ylabel("Clustering Coefficient",fontsize=16)
    plt.xlabel("Number of Nodes",fontsize=16)