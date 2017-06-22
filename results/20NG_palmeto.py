from palmettopy.palmetto import Palmetto
import numpy as np

# palmetto.get_coherence(words, coherence_type="cv")
# The available coherence types are "ca", "cp", "cv", "npmi", "uci", and "umass".

palmetto = Palmetto()

# 20NG NMF Okapi LV

topics = [['god', 'christian', 'jesus', 'people', 'bible', 'life', 'christ', 'make', 'thing', 'question'],
['window', 'file', 'program', 'run', 'do', 'windows', 'problem', 'application', 'version', 'm'],
['game', 'team', 'play', 'player', 'hockey', 'fan', 'win', 'season', 'baseball', 'year'],
['clipper', 'key', 'chip', 'encryption', 'escrow', 'government', 'algorithm', 'crypto', 'netcom', 'sternlight'],
['dod', 'bike', 'car', 'article', 'ride', 'write', 'motorcycle', 'rid', 'drive', 'bmw'],
['geb', 'gordon', 'pitt', 'bank', 'n3jxp', 'dsl', 'chastity', 'cadre', 'skepticism', 'shameful'],
['fbi', 'atf', 'waco', 'fire', 'burn', 'batf', 'koresh', 'ranch', 'survivor', 'gun'],
['sale', 'sell', 'offer', 'price', 'condition', 'interested', 'include', 'shipping', 'buy', 'pay'],
['covington', 'mcovingt', 'uga', 'ai', 'georgia', 'michael', '30602', 'n4tmi', '0358', '7415'],
['space', 'nasa', 'gov', 'henry', 'pat', 'digex', 'orbit', 'prb', 'spencer', 'zoo'],
['israel', 'israeli', 'arab', 'jew', 'jake', 'jewish', 'bony1', 'bony', 'palestinian', 'livni'],
['mail', 'email', 'university', 'info', 'information', 'post', 'address', 'fax', 'internet', 'advance'],
['card', 'drive', 'video', 'monitor', 'bus', 'scsi', 'board', 'mac', 'controller', 'ram'],
['1', '2', '3', '4', '5', '6', '0', '8', '7', '10'],
['cramer', 'clayton', 'gay', 'optilink', 'homosexual', 'percentage', 'study', 'uunet', 'consent', 'pyramid'],
['tek', 'robert', 'bobbe', 'beauchaine', 'ico', 'vice', 'bob', 'sank', 'bronx', 'manhattan'],
['keith', 'cco', 'caltech', 'schneider', 'allan', 'livesey', 'jon', 'wpd', 'solntze', 'sgi'],
['sandvik', 'apple', 'kent', 'newton', 'ksand', 'alink', 'net', 'cheer', 'private', 'activity'],
['ac', 'uk', 'ohio', 'magnus', 'state', 'write', 'article', '44', 'subject', 'dc'],
['armenian', 'turkish', 'armenia', 'turk', 'argic', 'serdar', 'soviet', 'muslim', 'zuma', 'serum']]

for words in topics:
    print(words, ':', round(palmetto.get_coherence(words),2))

# 20NG NMF Okapi CVV
topics = [['god', 'christian', 'jesus', 'christ', 'christianity', 'church', 'life', 'faith', 'bible', 'people'],
['window', 'file', 'program', 'application', 'problem', 'version', 'subject', 'manager', 'image', 'graphic'],
['game', 'team', 'player', 'fan', 'season', 'hockey', 'baseball', 'year', 'playoff', 'league'],
['clipper', 'chip', 'key', 'encryption', 'escrow', 'algorithm', 'government', 'crypto', 'sternlight', 'nsa'],
['car', 'bike', 'dod', 'article', 'engine', 'bmw', 'subject', 'motorcycle', 'mile', 'wheel'],
['fbi', 'atf', 'survivor', 'fire', 'ranch', 'dividian', 'waco', 'burn', 'batf', 'compound'],
['mac', 'apple', 'computer', 'machine', 'power', 'monitor', 'modem', 'port', 'board', 'price'],
['israel', 'israeli', 'jew', 'arab', 'jewish', 'jake', 'palestinian', 'livni', 'occupied', 'beyer'],
['mail', 'university', 'fax', 'internet', 'information', 'email', 'address', 'info', 'advance', 'phone'],
['geb', 'gordon', 'bank', 'n3jxp', 'chastity', 'skepticism', 'intellect', 'article', 'patient', 'medical'],
['space', 'henry', 'spencer', 'pat', 'mission', 'prb', 'orbit', 'shuttle', 'moon', 'earth'],
['cramer', 'clayton', 'gay', 'homosexual', 'percentage', 'study', 'optilink', 'uunet', 'men', 'consent'],
['michael', 'covington', 'mcovingt', 'georgia', 'n4tmi', 'radio', 'amateur', 'athens', 'artificial', 'intelligence'],
['card', 'driver', 'video', 'monitor', 'diamond', 'color', 'bus', 'mode', 'graphic', 'vga'],
['drive', 'disk', 'scsi', 'hard', 'controller', 'ide', 'hd', 'floppy', 'problem', 'system'],
['sale', 'offer', 'condition', 'price', 'shipping', 'excellent', 'original', 'subject', 'manual', 'mail'],
['keith', 'schneider', 'allan', 'morality', 'livesey', 'jon', 'ryan', 'kmr4', 'moral', 'atheist'],
['armenian', 'turkish', 'armenia', 'muslim', 'turk', 'serdar', 'argic', 'genocide', 'soviet', 'today'],
['sandvik', 'kent', 'alink', 'cheer', 'private', 'activity', 'net', 'article', 'malcolm', 'mlee'],
['gun', 'people', 'law', 'time', 'state', 'thing', 'government', 'case', 'weapon', 'police']]

for words in topics:
    print(words, ':', round(palmetto.get_coherence(words),2))