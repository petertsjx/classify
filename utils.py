import pandas as pd
import re
import emoji
from zhon.hanzi import punctuation

def remove_emoji_simple(text):
    if pd.isna(text):
        return ''
    return emoji.replace_emoji(str(text), '')


def process_dataframe(df):
    # 清洗title和description
    df['clean_title'] = df['title'].apply(clean_text)
    df['clean_description'] = df['description'].apply(clean_text)

    # 合并文本并去重
    df['merged_text'] = df.apply(
        lambda x: ' '.join(set(filter(None, [x['clean_title'], x['clean_description']]))),
        axis=1
    )

    # 删除合并后为空的行
    df = df[df['merged_text'].str.len() > 0]

    # 删除临时列
    df = df.drop(['clean_title', 'clean_description'], axis=1)

    return df

def clean_text(text):
    if pd.isna(text):
        return ''
    # 转换为字符串
    text = str(text)

    # 去除表情符号
    text = remove_emoji_simple(text)
    # 去除网址
    text = re.sub(r'http\S+|www.\S+', '', text)
    # 去除数字
    text = re.sub(r'\d+', '', text)
    # 去除英文
    text = re.sub(r'[a-zA-Z]+', '', text)
    # 去除中文标点符号
    text = re.sub(f'[{punctuation}]', '', text)
    # 去除其他特殊字符和空白字符
    text = re.sub(r'[^\u4e00-\u9fff]+', '', text)

    return text.strip()




def process_full_test_data(path):
    df = pd.read_csv(path)
    return process_dataframe(df)

def read_answer_tag(text):
    """
    提取指定标签中的内容

    Args:
        tag: 开始标签，如 "<title>"
        text: 包含标签的文本

    Returns:
        匹配到的标签内容列表
    """
    # 构造结束标签
    tag = "<answer>"
    tag_start = re.escape("<answer>")
    tag_end = re.escape(tag.replace('<', '</'))

    # 正则表达式模式
    pattern = f"{tag_start}(.*?){tag_end}"

    # 使用非贪婪匹配
    result = re.findall(pattern, text, re.DOTALL)
    return result[0] if result else ""


#print(process_full_test_data("../../data/full_test_update.csv"))