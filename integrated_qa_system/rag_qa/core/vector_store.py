# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain.docstore.document import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
from document_process import *
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
from base import logger, Config

conf = Config()


# 定义 VectorStore 类，封装向量存储和检索功能
class VectorStore:
    """
    向量存储和检索类，封装了 Milvus 的操作和向量检索功能
    """
    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=conf.MILVUS_COLLECTION_NAME,
                 host=conf.MILVUS_HOST,
                 port=conf.MILVUS_PORT,
                 database=conf.MILVUS_DATABASE_NAME):
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 设置日志记录器
        self.logger = logger

        # ========== GPU 加速相关变量（懒加载，避免启动卡顿） ==========
        self.embedding_function = None
        self.dense_dim = None
        self.reranker = None

        # 初始化 Milvus 客户端，连接到指定主机和数据库
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()

    # 【GPU 版】加载嵌入模型（懒加载：第一次用的时候再加载）
    """
    更换GPU加速模型
    """
    def load_embedding_model(self):
        if self.embedding_function is None:
            self.embedding_function = BGEM3EmbeddingFunction(
                model_name_or_path='E:/agent_class/model/bge-m3',
                use_fp16=True,  # GPU 开启半精度，速度翻倍、省显存
                device="cuda",  # 使用 GPU 加速
                local_files_only=True  # 强制只读本地文件，跳过联网校验
            )
            self.dense_dim = self.embedding_function.dim["dense"]
            logger.info("BGE-M3 嵌入模型（GPU）加载完成")

    # 【GPU 版】加载重排模型（懒加载：第一次用的时候再加载）
    """
    使用懒加载，防止直接加载卡顿
    """
    def load_reranker_model(self):
        if self.reranker is None:
            self.reranker = CrossEncoder(
                "E:\\agent_class\\model\\bge-reranker-large",
                device="cuda",  # 使用 GPU 加速
                local_files_only=True  # 强制只读本地文件，跳过联网校验
            )
            logger.info("BGE-Reranker 重排模型（GPU）加载完成")

    # 类私有化方法
    """
    创建向量数据库和对应的集合表
    """
    def _create_or_load_collection(self):
        # 检查指定集合是否已经存在
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT16_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT16_VECTOR,
                             dim=self.dense_dim if self.dense_dim else 1024)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    # 定义方法，向向量数据库存储添加文档
    """
    将处理切分好的文本块，进行向量化存储到向量数据库中
    """
    def add_documents(self, documents):
        # 启动 GPU 加速
        self.load_embedding_model()
        # 提取所有文档的内容列表
        texts = [doc.page_content for doc in documents]
        # 使用 BGE-M3 嵌入函数生成文档的嵌入
        embeddings = self.embedding_function(texts)
        # 初始空列表，用于存储向量数据库中的数据
        data = []
        # 遍历所有文档，带上索引
        for i, doc in enumerate(documents):
            # 生成文档内容的MD5 哈希值，作为唯一的ID
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            # 初始化稀疏向量字典（Milvus要求的稀疏向量的格式）
            sparse_vector = {}
            # 获取第i行对应的稀疏向量
            row = embeddings["sparse"].getrow(i)
            # 获取稀疏向量的非零值的索引
            indics = row.indices
            # 获取稀疏向量的非零值
            values = row.data
            # 将索引和值进行配对，存储到字典中
            for idx, value in zip(indics, values):
                sparse_vector[idx] = value
            # 创建数据字典，包含所有字段
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            })
        if data:
            # 将数据批量写入向量数据库
            self.client.upsert(collection_name=self.collection_name, data=data)
            # 记录添加文档的日志
            logger.info(f"已添加 {len(data)} 个文档到集合 {self.collection_name}")

    # 定义方法，执行混合检索并重排序
    """
    将问题也进行拆分向量化，再对向量数据库中的稠密和稀疏向量进行检索
    """
    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        # 使用 BGE-M3 嵌入函数生成查询的嵌入
        query_embeddings = self.embedding_function([query])

        # 确保稠密向量是float32类型的numpy数组
        dense_vector = query_embeddings["dense"][0]

        # 获取查询的稠密向量
        dense_query_vector = dense_vector
        # 初始化查询的稀疏向量字典
        sparse_query_vector = {}
        # 获取查询稀疏向量的第 0 行数据
        row = query_embeddings["sparse"].getrow(0)
        # 获取稀疏向量的非零值索引
        indices = row.indices
        # 获取稀疏向量的非零值
        values = row.data
        # 将索引和值配对，填充稀疏向量字典
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value

        # 初始化过滤表达式，默认不过滤
        """
        对文本进行过滤，提前做好分类
        """
        filter_expr = f"source == '{source_filter}'" if source_filter else ""
        # 创建稠密向量搜索请求
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )
        # 创建稀疏向量搜索请求
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )

        # 创建加权排序器，稀疏向量权重 0.7，稠密向量权重 1.0
        ranker = WeightedRanker(1.0, 0.7)
        # 执行混合搜索，返回 Top-K 结果
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
        )[0]
        # 提取搜索结果，将样本中的基本信息封装成 Document 对象
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        # 从子块中提取去重的父文档
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        # 如果只有1个文档，直接返回跳过重排序
        if len(parent_docs) < 2:
            return parent_docs[:conf.CANDIDATE_M]
            # 如果有父文档，进行重排序
        if parent_docs:
            # 创建查询与文档内容的配对列表
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # 使用 BGE-Reranker 计算每个配对的得分
            scores = self.reranker.predict(pairs)
            # 根据得分从高到低排序文档
            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        # 如果没有父文档，返回空列表
        else:
            ranked_parent_docs = []

        # 返回前 k 个重排序后的文档
        return ranked_parent_docs[:conf.CANDIDATE_M]

    # 定义私有方法，从子块中提取去重的父文档
    """
    
    """
    def _get_unique_parent_docs(self, sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
        # 返回去重后的父文档列表
        return unique_docs

    # 定义私有方法，从 Milvus 查询结果创建 Document 对象
    """
    创建实例化对象，把得到的chunk中的基本信息封装到这个实例化对象当中
    """
    def _doc_from_hit(self, hit):
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )
if __name__ == '__main__':
    vector_store = VectorStore()
    directoy_path = "E:/ai-agent/EDUagent/integrated_qa_system/rag_qa/data"
    documents = process_documents(directoy_path)
    vector_store.add_documents(documents)
