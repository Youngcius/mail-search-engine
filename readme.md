- 运行环境

  ./src 目录下源文件在 Python 3.7.6 环境下调试通过

  依赖的主要Python库：nltk，numpy，pandas，scipy

- 运行说明

  ```bash
  # python bool_search.py -h，查看说明
  # python semantic_search -h，查看说明
  python bool_search # 运行布尔检索程序
  python semantic_search # 运行语义检索程序
  ```

  ./output中已经存放经过序列化的DocumentProcessed实例（其中包括倒排表、TF-iDF稀疏矩阵等属性），对象构造过程较为耗时，用户运行中可直接使用该存储对象

- 函数及类说明

  DocumentProcessed 类表示对原始数据目录中文件作分词、词根化处理过的文档类；主要属性及方法如下

  ```python
  self.data_path  # 所依赖的数据目录
  self.files_path  # 各个文档（邮件）的绝对路径名组成的列表（list）
  self.num_docs  # 文档总数（int）
  self.documents  # 各文档中包含的词条组成的嵌套列表（list）
  self.tf  # TF矩阵（CSC Sparse Matrix)
  self.df  # DF向量（dict）
  self.inverted_index  # 倒排表， e.g. 'yes':[1,2]
  self.tfidf = sparse.csc_matrix([])  # TF-iDF矩阵（CSC Sparse Matrix)
  
  self.process(in_path: str)  #  处理文档，构建除了倒排表、TF-iDF之外的其他属性
  self.inverted_index_create() # 构建倒排表
  self.tfidf_create() # 构建TF-iDF矩阵
  ```

  Retriever 检索器基类：包含init（初始化）、action（执行检索）、represent（显示检索结果）三个方法

  BoolRetriever 子类：主要实现根据DocumentProcessed类型实例（实为其中倒排表）以及query命令（str类型）执行检索和显示检索结果的方法

  SemanticRetriever 子类：主要实现根据DocumentProcessed类型实例（实为其中TD-iDF矩阵）以及query命令（str类型）执行检索和显示检索结果的方法

  用户使用说明，（e.g. 布尔检索）

  ```python
  # document = DocumentProcessed(data_path)
  # // or
  # with open(os.path.join(args.output_path, DOCUMENT_NAME), 'rb') as f:
  # 	document = pickle.load(f)
  cmd = 'meeting and company or (financial and not issues)'
  bool_retriever = BoolRetriever(query=cmd)
  bool_retriever.action(document=document)  # 解析检索命令, 执行检索算法, 返回检索结果
  bool_retriever.represent()  #检索结果展示
  ```

- 文件说明

  ./output 中已存储有基于前50000个文档、文档频率前1000的词条构建的DocumentProcessed类型的实例（二进制文件），用户运行默认存在该文件时直接读取，不必重新运行程序以构造

  ./dataset 中的 ../dataset/maildir/ 下为各个邮件文件的所在各个目录；如 ./dataset/maildir/brawner-s/deleted_items/243 为某邮件文件的路径名

- 数据集原始文件

  安然公司（Enron）员工往来通讯邮件，共计 50 余万个文本文件。

  已分享在USTC睿客网，链接：https://rec.ustc.edu.cn/share/01b6e370-3565-11eb-baf2-b73de2be9edb  密码：faai

