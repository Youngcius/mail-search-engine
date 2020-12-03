import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy import sparse
import warnings
import nltk
import pickle
import os
import argparse
from bool_search import Retriever, DocumentProcessed

DOCUMENT_NAME = 'document.pk'


class SemanticRetriever(Retriever):
    def __init__(self, query: str):
        super(SemanticRetriever, self).__init__(query)
        self.query_arr = np.array([])

    def action(self, document: DocumentProcessed):
        """
        :param document: 基于该DocumentProcessed类中的tfidf矩阵进行查询和排序
        """
        if self.searched:
            pass  # 已经生成查询结果的不用执行该方法
        else:
            self.searched = True
            # str类型query命令待解析为1-D ndarray类型
            query_list = self.query.split()

            # 词根化
            snowball = nltk.SnowballStemmer('english')
            query_list = [snowball.stem(w) for w in query_list]
            wnl = nltk.WordNetLemmatizer()
            query_list = [wnl.lemmatize(w) for w in query_list]

            # 向量化
            self.query_arr = np.zeros(document.tfidf.shape[0])
            for token in query_list:
                if token in document.token_dict.keys():
                    self.query_arr[document.token_dict[token]] = 1

            # 归一化向量，执行查询（内积），进行比较和排序
            tfidf_arr = document.tfidf.toarray()
            mat_norm = np.linalg.norm(tfidf_arr, axis=0)
            mat_norm[mat_norm == 0] = 1  # 列向量norm为0说明该列元素全为0
            tfidf_arr /= mat_norm
            self.query_arr /= np.linalg.norm(self.query_arr)

            result_vec = np.array([np.dot(tfidf_arr[:, i], self.query_arr) for i in range(document.tfidf.shape[1])])
            order = np.flip(result_vec.argsort())
            self.result = [document.files_path[i] for i in order]  # result是文档ID对应的文档路径名（list）

    def represent(self, num=10):
        """
        :param num: 最大显示数量
        """
        if len(self.result) == 0:
            print("Your query has not been executed!")
        else:
            print("Results (absolute file path) is/are:")
            for i in range(num):
                print(self.result[i])


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='Semantic search')
    parser.add_argument('-i', '--data_path', default='../data', type=str, help='set the data directory')
    parser.add_argument('-o', '--output_path', default='../output', type=str, help='set the output directory')
    parser.add_argument('-n', '--num_result', default=10, type=int, help='number of results (in order)')
    args = parser.parse_args()

    # 文档词条化预处理
    if DOCUMENT_NAME not in os.listdir(args.output_path):
        document = DocumentProcessed(in_path=args.data_path)
        with open(os.path.join(args.output_path, DOCUMENT_NAME), 'wb')as f:
            pickle.dump(document, f)
    else:
        with open(os.path.join(args.output_path, DOCUMENT_NAME), 'rb')as f:
            document = pickle.load(f)

    cmd = ''  # 输入检索命令 e.g. company meeting market electricity financial offer customers
    print("===============Bool retrieval system for mails===============")
    print("use ' ' separate your query command (tokens) ; input '#' to end search action")

    while True:
        cmd = input("Input your query command: ").lower()  # 小写化输入
        if cmd == '#':
            break
        semantic_retriever = SemanticRetriever(query=cmd)
        semantic_retriever.action(document=document)  # 解析检索命令, 执行检索算法, 返回检索结果
        semantic_retriever.represent(num=args.num_result)  # 检索结果展示

    print("Query has exited. Thank you for your performance!")
