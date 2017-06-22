from palmettopy.palmetto import Palmetto
import numpy as np

# palmetto.get_coherence(words, coherence_type="cv")
# The available coherence types are "ca", "cp", "cv", "npmi", "uci", and "umass".

palmetto = Palmetto()

# LDA TFIDF LV
topics = [
['problem', 'algorithm', 'bound', 'graph', 'time', '1', '2', 'approximation', 'show', 'log'],
['classification', 'learn', 'data', 'learning', 'rule', 'classifier', 'neural', 'base', 'network', 'discovery'],
['data', 'query', 'database', 'system', 'propose', 'paper', 'performance', 'base', 'present', 'show'],
['image', 'method', 'model', 'surface', 'base', '3d', 'motion', 'object', 'present', 'shape'],
['research', 'system', 'information', 'paper', 'technology', 'data', 'medical', 'application', 'knowledge', 'web']
]


for words in topics:
    print(words, ':', palmetto.get_coherence(words))

# LDA TFIDF CVV
topics = [
['problem', 'algorithm', 'graph', 'time', 'bound', 'approximation', 'polynomial', 'number', 'log', 'result'],
['data', 'mining', 'algorithm', 'method', 'paper', 'pattern', 'rule', 'set', 'approach', 'tree'],
['medical', 'clinical', 'patient', 'network', 'neural', 'research', 'image', 'brain', 'computer', 'knowledge'],
['image', 'visualization', 'surface', 'method', 'model', 'motion', 'object', 'shape', 'point', 'volume'],
['system', 'database', 'data', 'query', 'information', 'application', 'management', 'paper', 'user', 'language'],
]

for words in topics:
    print(words, ':', palmetto.get_coherence(words))