import spacy
import pandas as pd
from collections import defaultdict
import jieba
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
from spacy.language import Language
from typing import List, Dict, Any


class ChineseVocabularyManager:
    def __init__(self, model='zh_core_web_sm'):
        """
        初始化 SpaCy 中文模型和词汇管理器

        参数:
        - model: SpaCy 中文模型名称
        """
        try:
            # 加载预训练模型
            self.nlp = spacy.load(model)

            # 禁用不需要的管道
            self.nlp.disable_pipes(['parser', 'ner'])
        except OSError:
            # 如果模型未下载，提供下载指导
            print(f"模型 {model} 未找到，请先下载:")
            print("python -m spacy download zh_core_web_sm")
            self.nlp = None

    def add_custom_vocabulary(self, words: List[str]):
        """
        添加自定义词汇到 SpaCy 词典

        参数:
        - words: 自定义词汇列表
        """
        if not self.nlp:
            print("模型未初始化")
            return

        # 使用 Matcher 添加词汇
        for word in words:
            self.nlp.vocab.strings.add(word)

    def add_entity_recognition(self, entities: Dict[str, str]):
        """
        添加自定义命名实体识别

        参数:
        - entities: 实体字典，格式为 {实体: 实体类型}
        """
        if not self.nlp:
            print("模型未初始化")
            return

        # 创建自定义实体识别规则
        @Language.component("custom_entity_recognizer")
        def custom_entity_recognizer(doc):
            # 创建 Span 列表（按照实体长度降序排序，避免重叠）
            matches = []
            sorted_entities = sorted(entities.items(), key=lambda x: len(x[0]), reverse=True)

            for entity_text, entity_label in sorted_entities:
                # 使用更安全的方法查找实体
                start_index = doc.text.find(entity_text)
                if start_index != -1:
                    # 使用 doc.char_span 并添加容错处理
                    span = doc.char_span(
                        start_index,
                        start_index + len(entity_text),
                        label=entity_label
                    )

                    # 检查 span 是否有效，并且不与已存在的实体重叠
                    if span and not any(
                            span.start < existing.end and span.end > existing.start for existing in matches):
                        matches.append(span)

            # 替换原有实体（而不是追加）
            doc.ents = matches
            return doc

        # 添加组件到流程
        self.nlp.add_pipe("custom_entity_recognizer", last=True)

    def custom_tokenization(self, text: str) -> List[str]:
        """
        自定义分词方法
        结合 SpaCy 和 Jieba 分词

        参数:
        - text: 输入文本

        返回:
        分词结果列表
        """
        # Jieba 精确分词
        jieba_tokens = list(jieba.cut(text, cut_all=False))

        # SpaCy 处理
        doc = self.nlp(text)
        spacy_tokens = [token.text for token in doc]

        # 合并去重
        combined_tokens = list(dict.fromkeys(jieba_tokens + spacy_tokens))
        return combined_tokens

    def word_frequency_analysis(self, text: str) -> Dict[str, int]:
        """
        词频分析

        参数:
        - text: 输入文本

        返回:
        词频统计字典
        """
        doc = self.nlp(text)

        # 词频统计
        word_freq = {}
        for token in doc:
            # 过滤停用词和标点
            if not token.is_stop and not token.is_punct:
                word_freq[token.text] = word_freq.get(token.text, 0) + 1

        return dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))

    def print_token_details(self, text: str):
        """
        打印分词和实体详细信息

        参数:
        - text: 输入文本
        """
        doc = self.nlp(text)

        print("分词和实体详细信息:")
        # 打印分词信息
        for token in doc:
            print(f"文本: {token.text}")
            print(f"词性: {token.pos_}")

        # 打印实体信息
        print("\n实体信息:")
        for ent in doc.ents:
            print(f"实体: {ent.text}, 类型: {ent.label_}")



# 加载中文模型
try:
    nlp = spacy.load("zh_core_web_trf")  # 转换器模型，准确率更高
except:
    nlp = spacy.load("zh_core_web_sm")  # 如果没有安装转换器模型，使用小模型


def create_entity_map():
    """创建实体类型映射字典"""
    return {
        'PERSON': '[人名]',
        'GPE': '[地点]',
        'LOC': '[地点]',
        'ORG': '[组织]',
        'DATE': '[日期]',
        'TIME': '[时间]',
        'MONEY': '[金额]',
        'PERCENT': '[百分比]',
        'QUANTITY': '[数量]',
        'CARDINAL': '[数值]',
        'FAC': '[设施]',
        'PRODUCT': '[产品]',
        'EVENT': '[事件]',
        'WORK_OF_ART': '[作品]',
        'LAW': '[法律]',
        'LANGUAGE': '[语言]'
    }


def extract_entities(text, nlp=nlp):
    """提取文本中的实体"""
    if pd.isna(text):
        return [], {}

    doc = nlp(str(text))
    entities = []
    entity_dict = defaultdict(list)

    for ent in doc.ents:
        entity = {
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        }
        entities.append(entity)
        entity_dict[ent.label_].append(ent.text)

    return entities, dict(entity_dict)


def generalize_entities(text, nlp=nlp):
    """实体泛化"""
    if pd.isna(text):
        return ''

    doc = nlp(str(text))
    entity_map = create_entity_map()

    # 按位置从后向前替换，避免位置变化
    replacements = []
    for ent in doc.ents:
        if ent.label_ in entity_map:
            replacements.append((ent.start_char, ent.end_char, entity_map[ent.label_]))

    # 从后向前替换
    text = list(text)
    for start, end, replacement in sorted(replacements, reverse=True):
        text[start:end] = replacement

    return ''.join(text)


def process_dataframe_with_ner(df, text_column):
    """处理DataFrame中的文本列"""
    # 实体提取
    entities_results = df[text_column].apply(extract_entities)
    df['entities'] = entities_results.apply(lambda x: x[0])
    df['entity_dict'] = entities_results.apply(lambda x: x[1])

    # 实体泛化
    df['generalized_text'] = df[text_column].apply(generalize_entities)

    return df


def main():
    # 创建词汇管理器
    vocab_manager = ChineseVocabularyManager()

    # 添加自定义词汇
    custom_words = [
        "人工智能",
        "大数据",
        "区块链",
        "机器学习"
    ]
    vocab_manager.add_custom_vocabulary(custom_words)

    # 添加自定义实体识别
    custom_entities = {
        "阿里巴巴": "ORG",
        "腾讯": "ORG",
        "百度": "ORG",
        "人工智能": "TECH"
    }
    vocab_manager.add_entity_recognition(custom_entities)

    # 测试文本
    test_text = "美国总统特朗普"

    # 自定义分词
    tokens = vocab_manager.custom_tokenization(test_text)
    print("分词结果:")
    print(tokens)

    # 词频分析
    word_freq = vocab_manager.word_frequency_analysis(test_text)
    print("\n词频分析:")
    print(word_freq)

    # 打印分词和实体详细信息
    vocab_manager.print_token_details(test_text)


# 更多高级实体识别示例
def advanced_entity_recognition_example():
    # 创建词汇管理器
    vocab_manager = ChineseVocabularyManager()

    # 复杂的实体识别场景
    complex_entities = {
        "中国": "COUNTRY",
        "人工智能": "TECH",
        "科技公司": "INDUSTRY",
        "北京": "CITY",
        "特朗普": "PERSON"
    }
    vocab_manager.add_entity_recognition(complex_entities)

    # 测试复杂文本
    complex_text = "美国总统特朗普"

    # 打印详细信息
    vocab_manager.print_token_details(complex_text)


if __name__ == "__main__":
    main()
    print("\n===== 高级实体识别示例 =====")
    advanced_entity_recognition_example()

"""
text = "美国总统特朗普"

# 提取实体
entities, entity_dict = extract_entities(text)
print("提取的实体:", entities)
print("实体字典:", entity_dict)
"""
"""

# 实体泛化
generalized = generalize_entities(text)
print("泛化后的文本:", generalized)

# 2. DataFrame处理
df = pd.DataFrame({
    'text': ['马云在杭州阿里巴巴宣布投资计划。', 
             '李明在北京召开新闻发布会。']
})
processed_df = process_dataframe_with_ner(df, 'text')
"""