import pyspark
import json

import re
from pattern.en import parse
from pattern.en import pprint
from pattern.vector import stem, PORTER, LEMMA
from nltk.corpus import stopwords

# adapted from HW5
def get_parts(thetext):
    # generate stopwords list & regexes for 2+ periods or 2+ dashes
    stop = stopwords.words('english')
    regex1=re.compile(r"\.{2,}")
    regex2=re.compile(r"\-{2,}")
    thetext=re.sub(regex1, ' ', thetext)
    thetext=re.sub(regex2, ' ', thetext)
    nouns=[]
    descriptives=[]
    for i,sentence in enumerate(parse(thetext, tokenize=True, lemmata=True).split()):
        nouns.append([])
        descriptives.append([])
        for token in sentence:
            if len(token[4]) >0:
                if token[1] in ['JJ', 'JJR', 'JJS']:
                    if token[4] in stop or token[4][0] in punctuation or token[4][-1] in punctuation or len(token[4])==1:
                        continue
                    descriptives[i].append(token[4])
                elif token[1] in ['NN', 'NNS']:
                    if token[4] in stop or token[4][0] in punctuation or token[4][-1] in punctuation or len(token[4])==1:
                        continue
                    nouns[i].append(token[4])
    out=zip(nouns, descriptives)
    nouns2=[]
    descriptives2=[]
    for n,d in out:
        if len(n)!=0 and len(d)!=0:
            nouns2.append(n)
            descriptives2.append(d)
    return nouns2, descriptives2

if __name__ == '__main__':
    # initialize Spark context
    conf = pyspark.SparkConf().setAppName("Twitter_Airline").setMaster("local[*]")
    sc = pyspark.SparkContext(conf=conf)

    # get tweets from text file
    # using sample tweets for now
    text_lines = sc.textFile('sample_tweets.json')
    tweets = text_lines.map(json.loads)
    tweets_text = tweets.map(lambda t: t['text'])

    # parse tweets to nouns & adjectives
    tweets_n_a = tweets_text.map(get_parts)
