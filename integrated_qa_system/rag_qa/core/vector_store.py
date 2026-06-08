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
    def load_reranker_model(self):
        if self.reranker is None:
            self.reranker = CrossEncoder(
                "E:\\agent_class\\model\\bge-reranker-large",
                device="cuda",  # 使用 GPU 加速
                local_files_only=True  # 强制只读本地文件，跳过联网校验
            )
            logger.info("BGE-Reranker 重排模型（GPU）加载完成")

    # 类私有化方法
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


if __name__ == '__main__':
    vector_store = VectorStore()
    directoy_path = "E:/ai-agent/EDUagent/integrated_qa_system/rag_qa/data"
    documents = process_documents(directoy_path)
    vector_store.add_documents(documents)
