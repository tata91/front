#!/usr/bin/python
# -*- coding: utf-8 -*-

import MySQLdb
import networkx as nx
import infomap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx.algorithms as nalgos
import re
import numpy as np
import pandas as pd
import lda
import nltk
import os

#链接数据库
def connect_mysql():
    conn = MySQLdb.connect(  host='localhost',
                             user='root',
                             passwd='001234',
                             auth_plugin='mysql_native_password',
                             db='complex_network',
                             charset='utf8')
    cursor = conn.cursor()
    return cursor


#建立文献标题和DOI的对应字典
def TIandDI(database_cursor):
    database_cursor.execute("select TI,DI from root_test")
    data = database_cursor.fetchall()
    dict_TIandDI = dict(data)
    return dict_TIandDI


#提取每个文献的参考文献并存储为list
def CRaslist(database_cursor):
    database_cursor.execute("select DI,CR from root_test")
    data = database_cursor.fetchall()
    data_refer = dict(data)
    for key in data_refer.keys():
        # 把每个文献的参考文献转为list
        #data_refer[key] = data_refer[key].split(';')
        data_refer[key] = re.split(';',data_refer[key])
    return data_refer


#从参考文献中提取DOI，用DOI代表文献
def DOIofCR(database_cursor):
    refer_data = CRaslist(database_cursor)
    for key in refer_data.keys():
        d = []
        for i in range(len(refer_data[key])):
            if 'DOI ' in refer_data[key][i]:
                refer_data[key][i] = refer_data[key][i].split('DOI ')[1].strip('[\'\']')
                d.append(refer_data[key][i])
            elif 'DOI' in refer_data[key][i]:
                refer_data[key][i] = refer_data[key][i].split('DOI')[1].strip('[\'\']')
                d.append(refer_data[key][i])
            else:
                continue
        refer_data[key] = d
    return refer_data

#值找键
def get_keys(d, value):
    return [k for k,v in d.items() if v == value]


#构建DOI和文献名对应词典
def DItoTI(database_cursor):
    database_cursor.execute("select DI,TI from root_test")
    data = database_cursor.fetchall()
    DItoTIdict = dict(data)
    return DItoTIdict

#提取文献出版年，并存储为字典
def YEARofCR(database_cursor):
    database_cursor.execute("select DI,PY from root_test")
    data = database_cursor.fetchall()
    YEAR_CR = dict(data)
    refer_data = CRaslist(database_cursor)
    refer_data_yearAndTI = {}
    for key in refer_data.keys():
        for i in range(len(refer_data[key])):
            k = 'null'
            v = 'null'
            j = refer_data[key][i]
            if 'DOI ' in refer_data[key][i]:
                refer_data[key][i] = refer_data[key][i].split('DOI ')[1]
                k = refer_data[key][i]
            elif 'DOI' in refer_data[key][i]:
                refer_data[key][i] = refer_data[key][i].split('DOI')[1]
                k = refer_data[key][i]
            else:
                continue
            #k = k.strip('[\'\']')
            j = j.split(',')
            v = j[1]
            refer_data_yearAndTI[k] = v
    #合并文献和参考文献的出版年对应字典
    YEAR_CR = {**YEAR_CR, **refer_data_yearAndTI}
    for key in YEAR_CR.keys():
        if type(YEAR_CR[key]) is str:
            if len(YEAR_CR[key]) > 5:
                YEAR_CR[key] = 0000
            else:
                YEAR_CR[key] = YEAR_CR[key][1:5]
        if YEAR_CR[key] is None:
            continue
        YEAR_CR[key] = int(YEAR_CR[key])
    return YEAR_CR

def deleteDuplicatedElementFromList(listA):
    # return list(set(listA))
    return sorted(set(listA), key=listA.index)


#搭建引用关系网络
def net_refer(database_cursor,begin_year,end_year):
    data_refer = DOIofCR(database_cursor)
    G = nx.Graph()
    for node in data_refer.keys():
        G.add_node(node)  # add a new node
        for key in data_refer.keys():
            for i in data_refer[key]:
                #print(data_refer[key])
                if i == node:
                    node2 = str(get_keys(data_refer,data_refer[key]))
                    G.add_edge(node,node2)
    #删除度为0的节点
    a = []
    for node in G.nodes:
        if G.degree(node) == 0:
            a.append(node)
    #删除不在指定出版年的节点
    year = YEARofCR(database_cursor)
    for node in G.nodes:
        node1 = node.strip('[\'\']')
        if year[node1] is None:
            a.append(node)
        elif year[node1]< begin_year:
            a.append(node)
        elif year[node1]>end_year:
            a.append(node)
        else:
            continue
    deleteDuplicatedElementFromList(a)

    if '10.1016/j.neuroimage.2009.10.003' in a:
        a.remove('10.1016/j.neuroimage.2009.10.003')

    G.remove_nodes_from(a)

    G_original = G
    #把节点标签换成数字，便于infomap算法计算
    num_nodes = G.number_of_nodes()
    mapping = dict(zip(G, range(1, num_nodes+1)))
    mapping_ = dict(zip(range(1, num_nodes + 1),G))
    G = nx.relabel_nodes(G, mapping)
    return G,mapping_,G_original






#搭建耦合关系网络,coupling(耦合)
def coupl_refer(database_cursor):
    referdata_list = CRaslist(database_cursor)
    listOfcouplingrefer = []
    print()

#infomap算法
def findcom_infomap(G):
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id and return number of communities found.
    """
    infomapWrapper = infomap.Infomap("--two-level")

    print("Building Infomap network from a NetworkX graph...")
    for e in G.edges():
        infomapWrapper.addLink(*e)

    print("Find communities with Infomap...")
    infomapWrapper.run()

    tree = infomapWrapper.clusterDataFile

    print("itertree with Infomap...")
    infomapWrapper.iterTree()

    print("Found %d modules with codelength: %f" % (infomapWrapper.numTopModules(), infomapWrapper.codelength()))

    communities = {}
    for node in infomapWrapper.iterLeafNodes():
        communities[node.physicalId] = node.moduleIndex()

    nx.set_node_attributes(G, name='community', values=communities)
    return infomapWrapper.numTopModules(),communities



#创建图
def createGraph(filename):
    file = open(filename, 'r')
    G = nx.Graph()
    list = []
    for line in file.readlines():
        nodes = line.split()
        edge = (int(nodes[0]), int(nodes[1]))
        G.add_edge(*edge)
    return G

'''
G,map,G1 = net_refer(cursor,2010,2010)
# infomap算法
mod_num, b = findcom_infomap(G)
#将文献用DOI表示
for key1 in map.keys():
    for key in b.keys():
        if key ==key1:
            b.update({map[key]: b.pop(key)})
list_p = []
for i in range(mod_num):
    i_list = []
    for key in b.keys():
        if b[key] == i:
            i_list.append(key)
    j = '10.1016/j.neuroimage.2009.10.003'
    if j in i_list:
        list_p = i_list
print(list_p)
'''

#提取摘要
def ABofDI(database_cursor):
    database_cursor.execute("select DI,AB from root_test")
    data_AB = database_cursor.fetchall()
    return data_AB

cursor = connect_mysql()
a = ABofDI(cursor)
a = dict(a)
dict_DIandTI = DItoTI(cursor)


G,map,G1 = net_refer(cursor,2011,2011)
# infomap算法
mod_num, b = findcom_infomap(G)
#将文献用DOI表示
for key1 in map.keys():
    for key in b.keys():
        if key ==key1:
            b.update({map[key]: b.pop(key)})
list_p = []
for i in range(mod_num):
    i_list = []
    for key in b.keys():
        if b[key] == i:
            i_list.append(key)
    j = '10.1016/j.neuroimage.2009.10.003'
    if j in i_list:
        list_p = i_list
print(len(list_p))






# 创建一个txt文件，文件名为mytxtfile,并向文件写入msg
def text_create(name, msg):
    desktop_path = "E:\\chinesearticle\\wenxianfenxi\\test\\1\\"  # 新创建的txt文件的存放路径
    full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')
    file.write(msg)   #msg也就是下面的Hello world!
    # file.close()


abs_set = []
for i in list_p:
    i = i.strip('[\'\']')
    k = dict_DIandTI[i]
    k = k.replace('?', '')

    str1= a[i]
    abs_set.append(str1)
    print(str1)
    text_create(k, str1)



    # print('分解后：')
    # tokens = nltk.word_tokenize(str)
    # stop_words = []
    # for line in open("E:/chinesearticle/wenxianfenxi/test/stop_words.txt", "r",encoding='utf-8'):
    #     line = line[:-1]
    #     stop_words.append(line)
    # filtered_sentence = [w for w in tokens if not w in stop_words]
    #
    #
    # print(filtered_sentence)










'''
cursor = connect_mysql()
for y in range(2009,2019):
    G,map,G1 = net_refer(cursor,y,y)
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print('这一年没有！！！')
        continue
    #print(map)
    #infomap算法
    mod_num,b = findcom_infomap(G)
    print('节点数：',end = '')
    print(G.number_of_nodes())
    print('{}年共有{}个社区，分别为：'.format(y, mod_num))

    #将文献用DOI表示
    for key1 in map.keys():
        for key in b.keys():
            if key ==key1:
                b.update({map[key]: b.pop(key)})
                #b[map[key1]] = b.pop(key)
    for i in range(mod_num):
        i_list = []
        for key in b.keys():
            if b[key] == i:
                i_list.append(key)
        for j in i_list:
            if j == '10.1016/j.neuroimage.2009.10.003':
                print('root_paper 在第{}社区'.format(i))
        num_list = len(i_list)
        print('第{}个社区包括{}个元素：'.format(i,num_list) , end='')
        print(i_list)
'''




#GN算法，太慢
# b1 = nalgos.community.girvan_newman(G1)
# top_level_communities = next(b1)
# next_level_communities = next(b1)









# for k, v in a.items():  # 遍历字典中的键值
#     s2 = str(v)  # 把字典的值转换成字符型
#     r.write(k + '\n')  # 键和值分行放，键在单数行，值在双数行
#     r.write(s2 + '\n')
#dic_new = dict(zip(map1.values(), map1.keys()))
#a,b = findcom_infomap(G)




#results = open("results3.txt", 'a')

#results.write(DOIofCR(cursor))
#df = pd.DataFrame(nx.to_numpy_matrix(G1))
#df.to_csv('results3.txt')
#G = createGraph('test.txt')


#print('visualize:',end=':')
#print(visualize(G))



#搭建共被引关系网络
#搭建作者合作网络
#文献共词分析:共关键词分析、共主题词分析
#文献网络裁剪技术




print('finish')
