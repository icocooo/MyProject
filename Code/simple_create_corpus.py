# simple_create_corpus.py
import pandas as pd
import os


def create_2gram_corpus_simple():
    """最简单的2-gram语料库创建"""

    # 1. 读取数据

    df = pd.read_csv('../Data/davis.csv')

    # 2. 创建数据目录
    os.makedirs('../Data', exist_ok=True)

    # 3. 创建蛋白质2-gram语料库
    with open('../Data/protein_corpus.txt', 'w') as f:
        for seq in df['target_sequence'].unique():
            # 2-gram分词：优先2字符，剩余用单字符
            grams = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq):
                    grams.append(seq[i:i + 2])  # 2-gram优先
                    i += 2
                else:
                    grams.append(seq[i])  # 单字符
                    i += 1
            f.write(' '.join(grams) + '\n')

    # 4. 创建SMILES 2-gram语料库
    with open('../Data/smiles_corpus.txt', 'w') as f:
        for smiles in df['compound_iso_smiles'].unique():
            grams = []
            i = 0
            while i < len(smiles):
                if i + 1 < len(smiles):
                    grams.append(smiles[i:i + 2])
                    i += 2
                else:
                    grams.append(smiles[i])
                    i += 1
            f.write(' '.join(grams) + '\n')

    print("语料库创建完成！")


if __name__ == '__main__':
    create_2gram_corpus_simple()