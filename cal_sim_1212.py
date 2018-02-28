import pymysql
import json
"""
     原始数据，用于建立模型
"""
# 缩水版的courses，实际数据的格式应该为 课程名\t课程简介\t课程详情，并已去除html等干扰因素
def mysql_run_sql(sql):
    db = pymysql.connect(
        host='172.18.126.51',
        port=3306,
        user='root',
        password='Abcd1234',
        database='math_compute',
        charset='utf8',
    )
    cursor = db.cursor()
    cursor.execute(sql)
    data = cursor.fetchall()
    # # 关闭数据库连接
    db.close()
    return data
courses_raw = mysql_run_sql("SELECT c_content FROM math_compute.news_result_02;")
courses = sum(courses_raw, ())

# 只是为了最后的查看方便
# 实际的 courses_name = [course.split('\t')[0] for course in courses]
courses_name = courses

"""
    预处理(easy_install nltk)
"""
def pre_process_cn(courses, low_freq_filter=True):
    """
     简化的 中文+英文 预处理
        1.去掉停用词
        2.去掉标点符号
        3.处理为词干
        4.去掉低频词

    """
    import nltk
    import jieba.analyse
    from nltk.tokenize import word_tokenize

    texts_tokenized = []
    for document in courses:
        texts_tokenized_tmp = []
        for word in word_tokenize(document):  #nltk.sent_tokenize(text) #对文本按照句子进行分割;nltk.word_tokenize(sent) #对句子进行分词
            texts_tokenized_tmp += jieba.analyse.extract_tags(word, 10)
        """
        关键词提取   jieba.analyse.extract_tags(word, 10)
        在构建VSM向量空间模型过程或者把文本转换成数学形式计算中，你需要运用到关键词提取的技术，这里就再补充该内容，而其他的如词性标注、并行分词、获取词位置和搜索引擎就不再叙述了。
        基本方法：jieba.analyse.extract_tags(sentence, topK)
        需要先import jieba.analyse，其中sentence为待提取的文本，topK为返回几个TF/IDF权重最大的关键词，默认值为20。"""
        texts_tokenized.append(texts_tokenized_tmp)

    texts_filtered_stopwords = texts_tokenized

    # 去除标点符号
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    texts_filtered = [[word for word in document if not word in english_punctuations] for document in
                      texts_filtered_stopwords]

    # 词干化
    from nltk.stem.lancaster import LancasterStemmer #处理-ing\-ed\-s 等后缀的词语
    st = LancasterStemmer()
    texts_stemmed = [[st.stem(word) for word in document] for document in texts_filtered]
    #print(texts_stemmed)

    # 去除过低频词
    if low_freq_filter:
        all_stems = sum(texts_stemmed, []) #将列表的列表合成一个总的列表[['a','b'],['c','d']] —>['a','b','c','d']
        #print(all_stems)
        stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1) #all_stems.count(stem)输出列表中stem的个数
        #print(stems_once)
        texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]
    else:
        texts = texts_stemmed
    return texts


lib_texts = pre_process_cn(courses)

"""
    引入gensim，正式开始处理(easy_install gensim)
"""


def train_by_lda(lib_texts):
    """
        通过LSI模型的训练
    """
    from gensim import corpora, models, similarities

    # 为了能看到过程日志
    # import logging
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    dictionary = corpora.Dictionary(lib_texts)
    corpus = [dictionary.doc2bow(text) for text in
              lib_texts]  # doc2bow(): 将collection words 转为词袋，用两元组(word_id, word_frequency)表示
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # 拍脑袋的：训练topic数量为10的LSI模型
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=100)
    corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    #index = similarities.MatrixSimilarity(lsi[corpus])  # index 是 gensim.similarities.docsim.MatrixSimilarity 实例
    #lda.print_topics(2)
    topics = ""
    return (corpus_lda,dictionary, lda)


# 库建立完成 -- 这部分可能数据很大，可以预先处理好，存储起来
(corpus_lda,dictionary, lda) = train_by_lda(lib_texts)
temp_topic_dict = {}
f1 = open(r"C:\Users\Administrator\Desktop\word_topic.json",'w+')
for each in range(len(lda.print_topics(100))):
    temp = lda.print_topics(100)[each] 
    temp_topic_dict[temp[0]] = temp[1]
json.dump(temp_topic_dict,f1)
f1.close()

f = open(r"C:\Users\Administrator\Desktop\word.json",'w+')
temp_dict = {}
for doc in ragne(len(corpus_lda)): # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
     temp_dict[doc]  = corpus_lda[doc]
json.dump(doc,f)
f.close()

from gensim import corpora, models, similarities
course1 =['没有\它很相似']
lib_texts = pre_process_cn(course1)
dictionary = corpora.Dictionary(lib_texts)
corpus = [dictionary.doc2bow(text) for text in
              lib_texts]  # doc2bow(): 将collection words 转为词袋，用两元组(word_id, word_frequency)表示
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
doc_lda = lda[corpus_tfidf]
for doc in doc_lda: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    print(doc)


# 要处理的对象登场
target_courses = [u'没有']
target_text = pre_process_cn(target_courses, low_freq_filter=False)

# """
# 对具体对象相似度匹配
# """
#
# # 选择一个基准数据
# ml_course = target_text[0]
#
# # 词袋处理
# ml_bow = dictionary.doc2bow(ml_course)
#
# # 在上面选择的模型数据 lsi 中，计算其他数据与其的相似度
# ml_lsi = lsi[ml_bow]  # ml_lsi 形式如 (topic_id, topic_value)
# #sims = index[ml_lsi]  # sims 是最终结果了， index[xxx] 调用内置方法 __getitem__() 来计算ml_lsi
#
# # 排序，为输出方便
# #sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
#
# # 查看结果
# # print(sort_sims[0:10])  # 看下前10个最相似的，第一个是基准数据自身
# # print(courses_name[sort_sims[1][0]])  # 看下实际最相似的数据叫什么
# # print(courses_name[sort_sims[2][0]])  # 看下实际最相似的数据叫什么
# # print(courses_name[sort_sims[3][0]])  # 看下实际最相似的数据叫什么

