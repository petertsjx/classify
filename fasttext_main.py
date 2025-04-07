import os
import pandas as pd
import numpy as np
import jieba
import re
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FastTextNewsClassifier:
    def __init__(self, data_path=None, label_prefix='__label__'):
        """
        初始化FastText新闻分类器

        参数:
            data_path: 数据文件路径，支持CSV或Excel格式
            label_prefix: FastText标签前缀
        """
        self.data_path = data_path
        self.label_prefix = label_prefix
        self.model = None
        self.label_dict = None
        self.inv_label_dict = None
        self.stopwords = set()

    def load_data(self, text_column='content', label_column='category', encoding='utf-8'):
        """
        加载数据

        参数:
            text_column: 文本列名
            label_column: 标签列名
            encoding: 文件编码

        返回:
            加载的DataFrame
        """
        print(f"正在加载数据: {self.data_path}")
        file_ext = os.path.splitext(self.data_path)[1].lower()

        if file_ext == '.csv':
            df = pd.read_csv(self.data_path, encoding=encoding)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(self.data_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")

        # 确保数据中包含所需的列
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"数据缺少必要的列: {text_column} 或 {label_column}")

        # 删除文本或标签为空的行
        df = df.dropna(subset=[text_column, label_column])

        # 创建标签映射字典
        labels = df[label_column].unique()
        self.label_dict = {label: i for i, label in enumerate(labels)}
        self.inv_label_dict = {i: label for label, i in self.label_dict.items()}

        print(f"数据加载完成，共 {len(df)} 条记录")
        print(f"类别分布:\n{df[label_column].value_counts()}")

        return df

    def load_stopwords(self, stopwords_path=None):
        """
        加载停用词

        参数:
            stopwords_path: 停用词文件路径
        """
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set([line.strip() for line in f])
            print(f"加载了 {len(self.stopwords)} 个停用词")
        else:
            print("未提供停用词文件或文件不存在，将不使用停用词")

    def preprocess_text(self, text):
        """
        文本预处理

        参数:
            text: 输入文本

        返回:
            处理后的文本
        """
        if not isinstance(text, str):
            return ""

        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)

        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # 移除数字和标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)

        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        # 分词
        words = jieba.cut(text)

        # 移除停用词
        if self.stopwords:
            words = [word for word in words if word not in self.stopwords and len(word.strip()) > 0]
        else:
            words = [word for word in words if len(word.strip()) > 0]

        return ' '.join(words)

    def prepare_data(self, df, text_column='content', label_column='category',
                     test_size=0.2, random_state=42):
        """
        准备训练和测试数据

        参数:
            df: 数据DataFrame
            text_column: 文本列名
            label_column: 标签列名
            test_size: 测试集比例
            random_state: 随机种子

        返回:
            train_file, test_file: 训练和测试文件路径
        """
        print("开始数据预处理...")

        # 预处理文本
        tqdm.pandas(desc="预处理文本")
        df['processed_text'] = df[text_column].progress_apply(self.preprocess_text)

        # 移除预处理后为空的行
        df = df[df['processed_text'].str.strip().str.len() > 0].reset_index(drop=True)

        # 划分训练集和测试集
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df[label_column]
        )

        print(f"训练集：{len(train_df)} 条记录, 测试集：{len(test_df)} 条记录")

        # 保存为FastText格式文件
        train_file = 'fasttext_train.txt'
        test_file = 'fasttext_test.txt'

        with open(train_file, 'w', encoding='utf-8') as f:
            for _, row in train_df.iterrows():
                label = row[label_column]
                text = row['processed_text']
                f.write(f"{self.label_prefix}{label} {text}\n")

        with open(test_file, 'w', encoding='utf-8') as f:
            for _, row in test_df.iterrows():
                label = row[label_column]
                text = row['processed_text']
                f.write(f"{self.label_prefix}{label} {text}\n")

        print(f"训练数据已保存至: {train_file}")
        print(f"测试数据已保存至: {test_file}")

        # 保存原始测试集用于后续评估
        self.test_df = test_df

        return train_file, test_file

    def train_model(self, train_file, dim=100, epoch=10, lr=0.1, wordNgrams=2,
                    minCount=5, loss='softmax'):
        """
        训练FastText模型

        参数:
            train_file: 训练文件路径
            dim: 词向量维度
            epoch: 训练轮数
            lr: 学习率
            wordNgrams: 词组长度
            minCount: 最小词频
            loss: 损失函数类型

        返回:
            训练好的模型
        """
        print(f"开始训练模型，参数: dim={dim}, epoch={epoch}, lr={lr}, wordNgrams={wordNgrams}")

        # 训练模型
        self.model = fasttext.train_supervised(
            train_file,
            dim=dim,
            epoch=epoch,
            lr=lr,
            wordNgrams=wordNgrams,
            minCount=minCount,
            loss=loss,
            label=self.label_prefix
        )

        # 保存模型
        model_path = 'fasttext_news_model.bin'
        self.model.save_model(model_path)
        print(f"模型已保存至: {model_path}")

        return self.model

    def evaluate_model(self, test_file):
        """
        评估模型性能

        参数:
            test_file: 测试文件路径

        返回:
            准确率, 召回率, F1分数
        """
        if self.model is None:
            raise ValueError("模型尚未训练")

        print("开始评估模型...")

        # 使用FastText内置评估
        n_samples,precision, recall = self.model.test(test_file)
        print(f"FastText内置评估 - 准确率: {precision:.4f}, 召回率: {recall:.4f}")

        # 自定义评估
        true_labels = []
        texts = []

        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        label, text = parts
                        label = label.replace(self.label_prefix, '')
                        true_labels.append(label)
                        texts.append(text)

        # 预测
        pred_labels = []
        for text in texts:
            label = self.model.predict(text)[0][0].replace(self.label_prefix, '')
            pred_labels.append(label)

        # 计算分类报告
        report = classification_report(true_labels, pred_labels)
        print("详细分类报告:")
        print(report)

        # 返回评估结果
        return precision, recall, report

    def predict(self, text, k=1):
        """
        预测单个文本的类别

        参数:
            text: 输入文本
            k: 返回的预测结果数量

        返回:
            预测的标签和概率
        """
        if self.model is None:
            raise ValueError("模型尚未训练")

        # 预处理文本
        processed_text = self.preprocess_text(text)

        # 预测
        labels, probs = self.model.predict(processed_text, k=k)

        # 处理结果
        results = []
        for label, prob in zip(labels, probs):
            label = label.replace(self.label_prefix, '')
            results.append((label, prob))

        return results

    def predict_batch(self, texts):
        """
        批量预测文本的类别

        参数:
            texts: 文本列表

        返回:
            预测的标签列表
        """
        if self.model is None:
            raise ValueError("模型尚未训练")

        results = []
        for text in tqdm(texts, desc="批量预测"):
            processed_text = self.preprocess_text(text)
            label = self.model.predict(processed_text)[0][0].replace(self.label_prefix, '')
            results.append(label)

        return results

    def visualize_confusion_matrix(self, true_labels, pred_labels):
        """
        可视化混淆矩阵

        参数:
            true_labels: 真实标签列表
            pred_labels: 预测标签列表
        """
        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels)

        # 获取唯一标签
        unique_labels = sorted(set(true_labels) | set(pred_labels))

        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=unique_labels,
                    yticklabels=unique_labels)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

        print(f"混淆矩阵已保存至: confusion_matrix.png")

    def load_model(self, model_path):
        """
        加载已训练的模型

        参数:
            model_path: 模型文件路径

        返回:
            加载的模型
        """
        if not os.path.exists(model_path):
            raise ValueError(f"模型文件不存在: {model_path}")

        self.model = fasttext.load_model(model_path)
        print(f"模型已从 {model_path} 加载")

        return self.model

    def run_full_pipeline(self, data_path, text_column='content', label_column='category',
                          stopwords_path=None, test_size=0.2, dim=100, epoch=10,
                          lr=0.1, wordNgrams=2):
        """
        运行完整的分类流程

        参数:
            data_path: 数据文件路径
            text_column: 文本列名
            label_column: 标签列名
            stopwords_path: 停用词文件路径
            test_size: 测试集比例
            dim: 词向量维度
            epoch: 训练轮数
            lr: 学习率
            wordNgrams: 词组长度
        """
        # 设置数据路径
        self.data_path = data_path

        # 加载数据
        df = self.load_data(text_column, label_column)

        # 加载停用词
        self.load_stopwords(stopwords_path)

        # 准备数据
        train_file, test_file = self.prepare_data(df, text_column, label_column, test_size)

        # 训练模型
        self.train_model(train_file, dim, epoch, lr, wordNgrams)

        # 评估模型
        self.evaluate_model(test_file)

        # 从测试集中提取真实标签和文本
        true_labels = self.test_df[label_column].tolist()
        texts = self.test_df[text_column].tolist()

        # 预测
        pred_labels = self.predict_batch(texts)

        # 可视化混淆矩阵
        #self.visualize_confusion_matrix(true_labels, pred_labels)

        # 返回测试结果
        return true_labels, pred_labels, texts


# 使用示例
if __name__ == "__main__":
    # 假设您的数据格式是这样的：
    # data = {
    #     'content': ['这是一条体育新闻...', '这是一条政治新闻...', ...],
    #     'category': ['体育', '政治', ...]
    # }
    # pd.DataFrame(data).to_csv('news_data.csv', index=False, encoding='utf-8')

    # 创建分类器实例
    classifier = FastTextNewsClassifier()

    # 运行完整流程
    true_labels, pred_labels, texts = classifier.run_full_pipeline(
        data_path='annotate_complete.csv',  # 您的数据文件路径
        text_column='content',  # 文本列名
        label_column='category',  # 标签列名
        stopwords_path='cn_stopwords.txt',  # 可选的停用词文件路径
        test_size=0.2,  # 测试集比例
        dim=200,  # 词向量维度
        epoch=20,  # 训练轮数
        lr=0.2,  # 学习率
        wordNgrams=2  # 词组长度
    )

    # 示例：预测新文本
    new_texts = [
        "中国足球队在世界杯预选赛中获胜",
        "国家主席访问欧洲多国进行外交活动",

    ]

    print("\n新文本预测示例:")
    for text in new_texts:
        result = classifier.predict(text, k=1)
        print(f"文本: {text}")
        print(f"预测类别: {result[0][0]}, 置信度: {result[0][1]:.4f}\n")