from utils import *
from spacy_entities import process_dataframe_with_ner
from chat import ChatBot
from tqdm import tqdm
from utils import read_answer_tag

def prompt_start():
    return '''
    我是数据标注工程师，现在要做一个中文新闻分类任务，你需要把新闻文本分成四类，分别是：
    1 国际时政
    2 金融财经
    3 web3(加密货币)
    4 国内事件
    如果不属于以上类别，请回答5，并自行给出一个分类。
    以下是需要进行分类的新闻，请直接写出答案，把答案放在标签<answer></answer>之间
    \n
    '''

data_path ="../../data/processed_full_results.csv"


def chat_with_bot():
    text = "基于把转为的一款工具智能提取精确识别文本表格可以保留文档的层级结构样式支持多页支持本地部署转"
    bot = "deepseek-r1:14b"
    prompt = prompt_start()
    chat_bot = ChatBot(prompt, bot)
    df = pd.read_csv(data_path)
    with open('classify_result.txt', 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            res = chat_bot.run([row['merged_text']])
            f.write(row['merged_text']+"_!_"+str(res[0])+"\n")
    return


def analyze_label_distribution(df):
    """
    分析标签分布情况

    Args:
        df: 包含label列的DataFrame
    """
    # 统计每个类别的数量
    label_counts = df['category'].value_counts()

    # 计算每个类别的占比
    label_percentages = df['category'].value_counts(normalize=True) * 100

    print("各类别数量统计:")
    print("-" * 20)
    for label in label_counts.index:
        print(f"类别 {label}: {label_counts[label]} 条 ({label_percentages[label]:.2f}%)")

    # 返回统计结果，以便进一步使用
    return label_counts, label_percentages

def process():
    data_path="classify_result.txt"
    data = pd.read_csv(data_path, sep="_!_", names=['text', 'label'])
    data['final_label'] = data['label'].apply(read_answer_tag)
    data[['final_label','text']].to_csv("./annotate_complete.csv",index=False,encoding='utf-8')


def main():
    data_an = "./annotate_complete.csv"
    df = pd.read_csv(data_an, encoding="utf-8")
    label_counts,label_percentages=analyze_label_distribution(df)
    print(label_counts,label_percentages)






if __name__ == "__main__":
    main()