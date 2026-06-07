# 这个脚本讲义的代码架构图没有体现，需要进行补充
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from datetime import datetime
import sys

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
# print(f'rag_qa_path--》{rag_qa_path}')
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
from edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from edu_text_spliter import ChineseRecursiveTextSplitter
from base import logger, Config

# 加载配置
conf = Config()

##todo：枚举文件类型，创建加载器字典
# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}


###todo：加载多种类型的文件并添加数据
def load_documents_from_directory(directory_path):
    # 创建一个空列表，用于存储加载的文档
    documents = []
    # 获取支持文件的扩展名集合
    supported_extensions = document_loaders.keys()
    # 从目录名提取学科类别（如"ai_data->ai"）为分类做铺垫
    # replace("_data", "") 这个地方是将_data替换为空
    source = os.path.basename(directory_path).replace("_data", "")
    # 遍历目录及其子目录
    for root, _, files in os.walk(directory_path):  # root 是目录名, files 是目录下的文件名
        logger.info(f"正在处理目录：{root}")
        # 构建完整的文件路径
        for file in files:
            file_path = os.path.join(root, file)
            # 获取文件的扩展名并转换为小写
            file_extension = os.path.splitext(file_path)[1].lower()
            # 检查文件扩展名是否在支持的扩展名列表中
            if file_extension in supported_extensions:
                try:
                    # 根据文件类型创建加载器实例
                    loader_class = document_loaders[file_extension]
                    # 特殊处理，当文件为txt文档时，要将文件编码改为utf-8
                    if file_extension == ".txt":
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)
                    # 调用加载器加载文档内容，返回文档列表
                    loaded_docs = loader.load()
                    # 遍历加载的每个文档
                    for doc in loaded_docs:
                        # 为文档添加学科类别元数据
                        doc.metadata["source"] = source
                        # 为文档添加文件路径元数据
                        doc.metadata["file_path"] = file_path
                        # 为文档添加当前时间戳元数据
                        doc.metadata["timestamp"] = datetime.now().isoformat()
                    logger.info(f"加载的文档数：{len(loaded_docs)}")
                    # 将加载的文档添加到总列表中
                    documents.extend(loaded_docs)
                    # 记录成功加载文档
                    logger.info(f"成功加载文件: {file_path}")
                except Exception as e:
                    # 记录错误
                    logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            else:
                # 记录警告
                logger.warning(f"不支持的文件类型: {file_path}")
    # 返回加载的文档
    print(f"加载后的文档：{documents}")
    return documents


# todo：将加载的文档进行分块
# 定义函数，处理文档并进行分层切分，返回子块结果
def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,  # 父块大小
                      child_chunk_size=conf.CHILD_CHUNK_SIZE,  # 子块大小
                      chunk_overlap=conf.CHUNK_OVERLAP):  # 块重叠
    # 从指定目录加载文档
    documents = load_documents_from_directory(directory_path)
    # 记录加载的文档总数日志
    logger.info(f"加载的文档总数：{len(documents)}")
    ###todo：初始化分词器
    # 初始化父块和子块分词器（通用。。。效果较慢）
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 初始化 Markdown 专用分词器 (效果很好)
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # 初始化一个空列表，用于存储子块
    child_chunks = []
    # 遍历每个原始文档，带上索引i
    for i, doc in enumerate(documents):
        # 获取文件的扩展名
        file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()
        # 选择分词器
        is_markdown = (file_extension == ".md")
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        logger.info(f"处理文档：{doc.metadata['file_path']},使用切分器{'Markdown' if is_markdown else '通用'}")

        # 将原始文档切分成父块
        parent_docs = parent_splitter_to_use.split_documents([doc])
        # 遍历父块，带上索引j
        for j, parent_doc in enumerate(parent_docs):
            # 为父块生成唯一 ID，格式为 "doc_i_parent_j"
            parent_id = f"doc_{i}_parent_{j}"
            # # 将父块 ID 添加到元数据
            # parent_doc.metadata["parent_id"] = parent_id
            # # 将父块内容存储到元数据
            # parent_doc.metadata["parent_content"] = parent_doc.page_content
            # 使用子块分词器将父块切分为子块
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])
            # 遍历子块，带上索引k
            for k, sub_chunk in enumerate(sub_chunks):
                # 为子块添加父块id到元数据中
                sub_chunk.metadata["parent_id"] = parent_id
                # 为子块添加父块内容到元数据中
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                # 为子块生成唯一 ID，格式为 "parent_id_child_k"
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                # 将子块添加到总列表中
                child_chunks.append(sub_chunk)
    logger.info(f"处理后的子块数：{len(child_chunks)}")
    return child_chunks
if __name__ == '__main__':
    # load_documents_from_directory("E:/ai-agent/EDUagent/integrated_qa_system/rag_qa/data")
    path = "E:/ai-agent/EDUagent/integrated_qa_system/rag_qa/data"
    process_documents(path)
