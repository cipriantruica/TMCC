# coding: utf-8

__author__ = "Ciprian-Octavian Truică"
__copyright__ = "Copyright 2017, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "ciprian.truica@cs.pub.ro"
__status__ = "Production"


import pymongo
import math
import numpy as np
from scipy.sparse import csr_matrix
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import normalize
from time import time

class MarketMatrix:
    def __init__(self, dbname='TwitterDB', parallel = 1):
        client = pymongo.MongoClient()
        self.dbname = dbname
        self.db = client[self.dbname]
        self.doc_cursor = None
        self.voc_cursor = None
        self.words  = {}
        self.documents = {}
        self.id2word = {}
        self.word2id = {}
        self.id2docid = {}
        self.docid2id = {}
        self.num_rows = 0
        self.num_columns = 0
        self.avgDocLen = 0
        self.parallel = parallel
        self.doc2class = {}

    """
        input:
            all: if True then use vocabulary_query, if False use the entire vocabulary
            limit: parameter used to limit the numeber of returned line, based on idf
            query: if all=True is used to select the documents for the query
    """
    def prepareData(self, all=True, query={}):
        if all:
            self.voc_cursor = self.db.vocabulary.find(no_cursor_timeout=True)
            self.doc_cursor = self.db.documents.find(no_cursor_timeout=True)
        else:
            self.voc_cursor = self.db.vocabulary_query.find(no_cursor_timeout=True)
            self.doc_cursor = self.db.documents.find(query, no_cursor_timeout=True)
        
        # parallel 
        if self.parallel == 1:
            print ("TO DO")
        elif self.parallel == 0: # single thread 
            print ('Single thread')
            self.avgDocLen = 0

            print ("Start vocabulary")
            idx = 0 
            for elem in self.voc_cursor:
                self.id2word[idx] = elem["word"]
                self.word2id[elem["word"]] = idx
                self.words[elem["word"]] = {
                    "IDF": elem["IDF"], 
                    "GTF": elem["GTF"], 
                    "SIDF": elem["SIDF"], 
                    "PIDF": elem["PIDF"], 
                    "SPIDF": elem["SPIDF"]
                }
                idx += 1
            print ("Finish vocabulary")

            self.num_columns = idx - 1

            print ("Start documents")
            idx = 0
            for elem in self.doc_cursor:
                d = {}
                docLen = 0
                for w in elem["words"]:
                    d[self.word2id[w["word"]]] = {
                        "TF": round(w["tf"], 2),  # TF double normalized
                        "count": round(w["count"], 2), # TF raw frequency
                        "NTF": round(1 + math.log(w["count"]), 2), # TF normalized
                        "IDF": round(self.words[w["word"]]["IDF"], 2), # IDF
                        "GTF": round(self.words[w["word"]]["GTF"], 2), # GTF the number of documents where the term appears
                        "SIDF": round(self.words[w["word"]]["SIDF"], 2),  # IDF smooth
                        "PIDF": round(self.words[w["word"]]["PIDF"], 2),  # IDF probabilistic
                        "SPIDF": round(self.words[w["word"]]["SPIDF"], 2) # IDF probabilistic smooth
                    }
                    docLen += w["count"]
                if d:
                    self.id2docid[idx] = elem["_id"]
                    self.docid2id[elem["_id"]] = idx
                    self.doc2class[idx] = elem["tags"][0]
                    self.documents[idx] = { "words": d, "docLen": docLen }
                    self.avgDocLen += docLen
                    idx += 1
            print ("Finish documents")
            self.num_rows = idx - 1
            self.avgDocLen /= self.num_rows
        # this is just for testing
        # for key, value in self.documents.items():
        #     print key, value

    # normalize a market matrix using l1, l2 or max norm on rows
    def normalize_MM(self, market_matrix, norm='l2', k=0.0):
        normalized_mm = []
        if norm == 'l1':
            for line in market_matrix:                
                l1_norm = sum([elem[1] for elem in line])
                if k < 0.5:
                    normalized_mm.append([(elem[0], elem[1]/l1_norm) for elem in line])
                else:
                    row = []
                    for elem in line:
                        new_elem = elem[1]/l1_norm
                        if new_elem < k and new_elem > 100-k:
                            new_elem = 0.0
                        row.append((elem[0], new_elem))
                    normalized_mm.append(row)
        if norm == 'l2':
            for line in market_matrix:
                # l2_norm = 0
                # for elem in line:
                    # l2_norm += elem[1] ** 2
                l2_norm = math.sqrt(sum([elem[1]**2 for elem in line]))
                # l2_norm = math.sqrt(l2_norm)
                if k < 0.5:
                    normalized_mm.append([(elem[0], elem[1]/l2_norm) for elem in line])
                else:
                    row = []
                    for elem in line:
                        new_elem = elem[1]/l2_norm
                        if new_elem < k and new_elem > 100-k:
                            new_elem = 0.0
                        row.append((elem[0], new_elem))
                    normalized_mm.append(row)
        if norm == 'max':
            # to do
            for line in market_matrix:
                max_norm = max([elem[1] for elem in line])
                if k < 0.5:
                    normalized_mm.append([(elem[0], elem[1]/max_norm) for elem in line])
                else:
                    row = []
                    for elem in line:
                        new_elem = elem[1]/max_norm
                        if new_elem < k and new_elem > 100-k:
                            new_elem = 0.0
                        row.append((elem[0], new_elem))
                    normalized_mm.append(row)
        return normalized_mm

    # normalize a csr matrix using l1, l2 or max norm on rows
    def normalize_CSR(self, csr, norm='l2'):
        normilized_csr = normalize(csr, norm=norm, axis=1, copy=True)
        return normilized_csr

    """
        constructs the binary market matrix
        output:
            the binary market matrix
    """
    def build_Binary_MM(self, filename=None):
        market_matrix = []
        data = []
        row = []
        col = []
        idx = 0
        for key in sorted(self.documents):
            l = []
            for word in sorted(self.documents[key]['words']):
                l.append((word, 1))
                row.append(idx)
                col.append(word)
                data.append(1)
            market_matrix.append(l)
            idx += 1
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)

        return market_matrix, csr

    """
        constructs the count market matrix
        output:
            the count market matrix
    """

    def build_Count_MM(self, filename=None):
        market_matrix = []
        data = []
        row = []
        col = []
        idx = 0

        for key in sorted(self.documents):
            l = []
            for word in sorted(self.documents[key]['words']):
                count = self.documents[key]['words'][word]['count']
                l.append((word, count))
                row.append(idx)
                col.append(word)
                data.append(count)
            market_matrix.append(l)
            idx += 1
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)

        return market_matrix, csr

    """
        constructs the TF market matrix
        output:
            the TF market matrix
    """
    def build_TF_MM(self, filename=None):
        market_matrix = []
        data = []
        row = []
        col = []
        idx = 0
        for key in sorted(self.documents):
            l = []
            for word in sorted(self.documents[key]['words']):
                tf = self.documents[key]['words'][word]['TF']
                l.append((word, tf))
                row.append(idx)
                col.append(word)
                data.append(tf)
            market_matrix.append(l)
            idx += 1
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)

        return market_matrix, csr


    """
        constructs the TF*IDF market matrix
        output:
            the TF*IDF market matrix
    """
    def build_TF_IDF_MM(self, filename=None):
        market_matrix = []
        data = []
        row = []
        col = []
        idx = 0
        for key in sorted(self.documents):
            l = []
            for word in sorted(self.documents[key]['words']):
                tf = self.documents[key]['words'][word]['TF']
                idf = self.documents[key]['words'][word]['IDF']
                tfidf = tf * idf
                l.append((word, tfidf))
                row.append(idx)
                col.append(word)
                data.append(tfidf)
            market_matrix.append(l)
            idx += 1
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)
        
        return market_matrix, csr

    """
        constructs the TF*SIDF market matrix
        output:
            the TF*SIDF market matrix
    """
    def build_TF_SIDF_MM(self, filename=None):
        market_matrix = []
        data = []
        row = []
        col = []
        idx = 0
        for key in sorted(self.documents):
            l = []
            for word in sorted(self.documents[key]['words']):
                tf = self.documents[key]['words'][word]['TF']
                idf = self.documents[key]['words'][word]['SIDF']
                tfidf = round(tf * idf, 2)
                l.append((word, tfidf))
                row.append(idx)
                col.append(word)
                data.append(tfidf)
            market_matrix.append(l)
            idx += 1
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)

        return market_matrix, csr

    """
        constructs the TF*PIDF market matrix
        output:
            the TF*PIDF market matrix
    """
    def build_TF_PIDF_MM(self, filename=None):
        market_matrix = []
        data = []
        row = []
        col = []
        idx = 0
        for key in sorted(self.documents):
            l = []
            for word in sorted(self.documents[key]['words']):
                tf = self.documents[key]['words'][word]['TF']
                idf = self.documents[key]['words'][word]['PIDF']
                tfidf = round(tf * idf, 2)
                l.append((word, tfidf))
                row.append(idx)
                col.append(word)
                data.append(tfidf)
            market_matrix.append(l)
            idx += 1
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)

        return market_matrix, csr

    """
        constructs the TF*SPIDF market matrix
        output:
            the TF*PIDF market matrix
    """
    def build_TF_SPIDF_MM(self, filename=None):
        market_matrix = []
        data = []
        row = []
        col = []
        idx = 0
        for key in sorted(self.documents):
            l = []
            for word in sorted(self.documents[key]['words']):
                tf = self.documents[key]['words'][word]['TF']
                idf = self.documents[key]['words'][word]['SPIDF']
                tfidf = round(tf * idf, 2)
                l.append((word, tfidf))
                row.append(idx)
                col.append(word)
                data.append(tfidf)
            market_matrix.append(l)
            idx += 1
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)
        
        return market_matrix, csr

    """
        constructs the Okapi BM25 (using TF*IDF) market matrix
        output:
            the Okapi BM25 market matrix
    """
    def build_Okapi_TF_IDF_MM(self, k1=1.6, b=0.75, filename=None):
        market_matrix = []
        data = []
        row = []
        col = []
        idx = 0
        for key in sorted(self.documents):
            l = []
            docLen = self.documents[key]['docLen']
            for word in sorted(self.documents[key]['words']):
                tf = self.documents[key]['words'][word]['TF']
                idf = self.documents[key]['words'][word]['IDF']
                okapi = (tf*idf*(k1+1))/(tf+k1*(1-b+b*docLen/self.avgDocLen))
                l.append((word, okapi))
                row.append(idx)
                col.append(word)
                data.append(okapi)
            market_matrix.append(l)
            idx += 1
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)

        return market_matrix, csr

    """
        constructs the Okapi BM25 (using TF*SPIDF) market matrix
        output:
            the Okapi BM25 market matrix
    """
    def build_Okapi_TF_SPIDF_MM(self, k1=1.6, b=0.75, filename=None):
        market_matrix = []
        data = []
        row = []
        col = []
        idx = 0
        for key in sorted(self.documents):
            l = []
            docLen = self.documents[key]['docLen']
            for word in sorted(self.documents[key]['words']):
                tf = self.documents[key]['words'][word]['TF']
                idf = self.documents[key]['words'][word]['SPIDF']
                okapi = (tf*idf*(k1+1))/(tf+k1*(1-b+b*docLen/self.avgDocLen))
                l.append((word, okapi))
                row.append(idx)
                col.append(word)
                data.append(okapi)
            market_matrix.append(l)
            idx += 1
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)
        

        return market_matrix, csr

# this are just for tests
def printMatGensim(mat, id2word):
    maximum = 0
    minimum = 100000.0
    for elem in mat:
        for id_p in elem:
            print( id2word[id_p[0]], id_p[1])
            if maximum < id_p[1]:
                maximum = id_p[1]
            if minimum > id_p[1]:
                minimum = id_p[1]
    print( maximum)
    print( minimum)

def printMatSklearn(mat, id2word):
    maximum = 0
    minimum = 100000.0
    for elem in mat.toarray():
        idx_col = 0
        for c in elem:
            if c > 0:
                print( id2word[idx_col], c)
                if maximum < c:
                    maximum = c
                if minimum > c:
                    minimum = c
            idx_col += 1
    print( maximum)
    print( minimum)
"""
# these are just tests
if __name__ == '__main__':
    start = time()
    mm = MarketMatrix(dbname='ConferenceDB', parallel=0)
    query = {"_id": {"$in": ["1", "2", "3", "4", "5"]}}
    mm.prepareData(all=False, query=query)
    # mm.prepareData(all=True)
    id2word = mm.id2word
    # m, c = mm.build_Count_MM()
    # print "MM Count"
    # printMatGensim(m)
    # print "CSR Count"
    # printMatSklearn(c)    
    # m, c = mm.build_TF_MM()
    # print "MM TF"
    # printMatGensim(m)
    # print "CSR TF"
    # printMatSklearn(c)
    # m, c = mm.build_TF_IDF_MM()
    # print "MM TF*IDF"
    # printMatGensim(mm.normalize_MM(m, 'l2'), id2word)
    # print "CSR TF*IDF"
    # printMatSklearn(mm.normalize_CSR(c, 'l2'), id2word)
    # m, c = mm.build_Okapi_TF_IDF_MM()
    # print "MM Okapi TF*IDF"
    # printMatGensim(m)
    # print "CSR Okapi TF*IDF"
    # printMatSklearn(c)
    
    # end = time()
    # print "Time:", (end - start)

    # mm = MarketMatrix(dbname='ConferenceDB', parallel=1)
    # mm.prepareData()
    # mm.build_Binary_MM()
    # print mm.avgDocLen

    # for given queries with/without limit
    # mm.build(query_exists=True)
    # mm.build(query=True, limit=100)
    # end = time()
    # print 'Build time:',(end-start)

    # start = time()
    # mm.buildBinaryMM('mm_binary.mtx')
    # end = time()
    # print "Binary MM time:", (end-start)

    # start = time()
    # mm.buildCountMM('mm_count.mtx')
    # end = time()
    # print "Binary Count time:", (end-start)

    # start = time()
    # id2word, id2tweetID, market_matrix = mm.buildTFMM('mm_tf.mtx')
    #
    # with codecs.open('id2word.txt', "w", "utf-8") as file1:
    #     file1.write("id2word\n")
    #     for elem in id2word:
    #         file1.write(str(elem) + " " + id2word[elem] + "\n")
    # file1.close()
    # with open('id2tweetID.txt' , 'w') as file2:
    #     file2.write("id2tweetID\n")
    #     for elem in id2tweetID:
    #         file2.write(str(elem) + " " + str(id2tweetID[elem]) + "\n")
    # file2.close()
    # end = time()
    # print "Binary TF time:", (end-start)
"""
