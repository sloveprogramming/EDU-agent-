# 导入标准库
import json
import os
# 导入 PyTorch
import torch
# 导入日志
from base import logger
# 导入numpy
import numpy as np
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

"""
初始化模型
"""


class QueryClassifier:
    def __init__(self, model_path="bert_query_classifier"):
        # 初始化模型路径
        self.model_path = model_path
        # 加载 BERT 分词器
        self.tokenizer = BertTokenizer.from_pretrained("E:\\agent_class\\model\\bert-base-chinese")
        # 初始化模型
        self.model = None
        # 确定设备（GPU 或 CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 记录设备信息
        logger.info(f"使用设备: {self.device}")
        # 定义标签映射
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        # 加载模型
        self.load_model()
    """
    加载模型
    """
    def load_model(self):
        # 检查本地模型是否存在
        if os.path.exists(self.model_path):
            # 从本地加载模型
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            logger.info(f"从本地加载模型: {self.model_path}")
        else:
            # 使用预训练的中文BERT模型
            self.model = BertForSequenceClassification.from_pretrained(
                "E:\\agent_class\\model\\bert-base-chinese",  # 使用已有的本地模型
                num_labels=2
            )
            self.model.to(self.device)
            logger.info(f"初始化新模型: {self.model_path}")


    """
    保存模型
    """
    def save_model(self):
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"模型保存至: {self.model_path}")


    """
    预处理数据BERT输出格式
    """
    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(
            texts,
            truncation=True,  # 超过max_length时截断
            padding="max_length",  # 不足max_length时填充
            max_length=128,  # 统一序列长度为128
            return_tensors="pt"  # 返回PyTorch张量格式
        )
        labels = [self.label_map[label] for label in labels]
        return encodings, labels


    """
    创建PyTorch 数据集
    """
    def create_dataset(self,encodings, labels):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self,encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item
        return Dataset(encodings, labels)


    """
    意图识别训练模型分类
    """
    def train_model(self, data_file="training_dataset_hybrid_5000.json"):
        # 加载数据集
        if not os.path.exists(data_file):
            logger.error(f"数据: {data_file}不存在")
            raise FileNotFoundError
        # 读取文件
        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f.readlines()]
        # 训练集的训练数据提取出来
        texts = [item["query"] for item in data]
        labels = [item["label"] for item in data]

        # 数据划分
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # 预处理
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)

        # 创建数据集
        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)

        # 设置训练参数
        training_args = TrainingArguments(
            output_dir="./bert_results",     # 模型输出目录
            num_train_epochs=3,              # 训练轮数
            per_device_train_batch_size=8,  # 每个设备（GPU/CPU）上训练时的批次大小，即每次前向传播处理的样本数
            per_device_eval_batch_size=8,   # 每个设备上评估时的批次大小
            warmup_steps=50,                # 学习率预热步数，在前50步训练中学习率从0逐渐增加到设定值，有助于稳定训练初期
            weight_decay=0.01,              # 权重衰减系数，用于 L2 正则化，防止模型过拟合
            logging_dir="./bert_logs",      # 日志和训练日志的输出目录
            logging_steps=10,               # 每训练10步记录一次日志信息（如损失、学习率等）
            eval_strategy="epoch",          # 评估策略，设置为 "epoch" 表示每个训练轮结束后进行一次评估
            save_strategy="epoch",          # 模型保存策略，设置为 "epoch" 表示每个训练轮结束后保存一次模型检查点
            load_best_model_at_end=True,    # 训练结束后自动加载训练过程中表现最好的模型
            save_total_limit=1,  # 只保存一个检查点,最多保存的模型检查点数量，超过此限制时会删除旧的检查点
            metric_for_best_model="eval_loss", # 	判断"最佳模型"的指标，这里使用评估损失（eval_loss），损失越小模型越好
            fp16=True,  # 	启用混合精度训练，使用 FP16（半精度）和 FP32（单精度）混合计算，可加速训练并减少显存占用
        )
        # 初始化Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
    """
    计算评估指标
    """
    def compute_metrics(self,eval_pred):\
        ...

if __name__ == '__main__':
    QC = QueryClassifier()
    QC.load_model()
    QC.train_model(data_file=os.path.join(
        os.path.dirname(__file__),
        "../classify_data/model_generic_5000.json"
    ))
