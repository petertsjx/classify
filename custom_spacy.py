import spacy
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc, Span
from spacy.matcher import PhraseMatcher
from spacy.language import Language
from typing import List, Dict, Tuple, Optional, Union


class CustomSpacy:
    def __init__(self, model_name: str = "zh_core_web_sm"):
        """
        初始化自定义Spacy类

        参数:
            model_name: 要加载的spacy模型名称
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"模型 {model_name} 未找到，正在下载...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)

        # 如果已存在EntityRuler，先移除
        if 'entity_ruler' in self.nlp.pipe_names:
            self.nlp.remove_pipe('entity_ruler')

        # 添加实体规则器
        self.entity_ruler = self.nlp.add_pipe('entity_ruler', before='ner')

        # 创建短语匹配器用于名词规则
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab)

        # 存储自定义的实体和名词规则
        self.custom_entities = []
        self.custom_noun_rules = []

        # 注册自定义名词处理器
        @Language.component("custom_noun_processor")
        def custom_noun_processor(doc):
            matches = self.phrase_matcher(doc)
            for match_id, start, end in matches:
                # 获取匹配的字符串和类型
                noun_type = self.nlp.vocab.strings[match_id]
                # 修改匹配到的token的POS标签
                for token_idx in range(start, end):
                    doc[token_idx].pos_ = noun_type
            return doc

        # 保存处理器引用
        self.custom_noun_processor = custom_noun_processor

    def add_entity(self, entity_text: str, entity_label: str) -> None:
        """
        添加单个实体规则

        参数:
            entity_text: 实体文本
            entity_label: 实体标签
        """
        pattern = {"label": entity_label, "pattern": entity_text}
        self.custom_entities.append(pattern)
        self.entity_ruler.add_patterns([pattern])

    def add_entities(self, entities: List[Dict[str, str]]) -> None:
        """
        批量添加实体规则

        参数:
            entities: 包含实体模式的字典列表，每个字典包含 'pattern' 和 'label'
        """
        self.custom_entities.extend(entities)
        self.entity_ruler.add_patterns(entities)

    def add_entities_from_list(self, entity_texts: List[str], entity_label: str) -> None:
        """
        从文本列表添加多个具有相同标签的实体

        参数:
            entity_texts: 实体文本列表
            entity_label: 所有实体共用的标签
        """
        patterns = [{"label": entity_label, "pattern": text} for text in entity_texts]
        self.custom_entities.extend(patterns)
        self.entity_ruler.add_patterns(patterns)

    def add_noun_rule(self, noun_text: str, noun_type: str = "NOUN") -> None:
        """
        添加名词规则

        参数:
            noun_text: 名词文本
            noun_type: 名词类型 (默认为NOUN)
        """
        # 将名词添加到自定义规则中
        self.custom_noun_rules.append((noun_text, noun_type))

        # 使用PhraseMatcher添加规则
        pattern = self.nlp(noun_text)
        self.phrase_matcher.add(noun_type, [pattern])

        # 添加自定义名词处理器（如果尚未添加）
        if 'custom_noun_processor' not in self.nlp.pipe_names:
            self.nlp.add_pipe('custom_noun_processor', after='tagger')

    def add_noun_rules(self, nouns: List[Tuple[str, str]]) -> None:
        """
        批量添加名词规则

        参数:
            nouns: 元组列表，每个元组包含 (名词文本, 名词类型)
        """
        for noun_text, noun_type in nouns:
            self.add_noun_rule(noun_text, noun_type)

    def process(self, text: str) -> Doc:
        """
        处理文本

        参数:
            text: 要处理的文本

        返回:
            处理后的Doc对象
        """
        return self.nlp(text)

    def get_entities(self, text: str) -> List[Tuple[str, str, str]]:
        """
        从文本中获取实体

        参数:
            text: 要处理的文本

        返回:
            实体列表，每个实体为一个元组 (文本, 标签, 位置)
        """
        doc = self.process(text)
        return [(ent.text, ent.label_, f"{ent.start_char}:{ent.end_char}") for ent in doc.ents]

    def get_nouns(self, text: str) -> List[Tuple[str, str]]:
        """
        从文本中获取所有名词

        参数:
            text: 要处理的文本

        返回:
            名词列表，每个名词为一个元组 (文本, 类型)
        """
        doc = self.process(text)
        return [(token.text, token.pos_) for token in doc if token.pos_ in ("NOUN", "PROPN")]

    def save_rules(self, file_path: str) -> None:
        """
        保存自定义规则到文件

        参数:
            file_path: 文件路径
        """
        import json
        rules = {
            "entities": self.custom_entities,
            "nouns": self.custom_noun_rules
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)

    def load_rules(self, file_path: str) -> None:
        """
        从文件加载自定义规则

        参数:
            file_path: 文件路径
        """
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)

        # 加载实体规则
        if "entities" in rules:
            self.add_entities(rules["entities"])

        # 加载名词规则
        if "nouns" in rules:
            self.add_noun_rules(rules["nouns"])


# 创建自定义Spacy实例
custom_nlp = CustomSpacy("zh_core_web_sm")

# 添加自定义实体
custom_nlp.add_entity("北京大学", "ORG")
custom_nlp.add_entity("科普法", "NOUN")
custom_nlp.add_entity("复旦大学", "ORG")
custom_nlp.add_entities_from_list(["上海", "北京", "广州"], "GPE")

# 添加自定义名词规则
custom_nlp.add_noun_rule("人工智能", "NOUN")
custom_nlp.add_noun_rule("深度学习", "NOUN")

# 处理文本
text = "修订后的科普法能带来哪些变化"
doc = custom_nlp.process(text)

# 获取实体
entities = custom_nlp.get_entities(text)
print("实体:", entities)

# 获取名词
nouns = custom_nlp.get_nouns(text)
print("名词:", nouns)