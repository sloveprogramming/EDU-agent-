import os
import sys

# 将 rag_qa 和项目根目录加入 sys.path，确保跨目录导入
# 与项目中 mysql_qa/retrieval/bm25_search.py 的 sys.path 模式保持一致
current_dir = os.path.dirname(os.path.abspath(__file__))   # rag_qa/core/
rag_qa_dir = os.path.dirname(current_dir)                  # rag_qa/
project_root = os.path.dirname(rag_qa_dir)                 # integrated_qa_system/
sys.path.insert(0, rag_qa_dir)
sys.path.insert(0, project_root)
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownTextSplitter
from datetime import datetime
from edu_text_spliter import AliTextSplitter, ChineseRecursiveTextSplitter
from edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from base import logger, Config
