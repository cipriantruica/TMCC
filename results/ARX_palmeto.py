from palmettopy.palmetto import Palmetto
import numpy as np

# palmetto.get_coherence(words, coherence_type="cv")
# The available coherence types are "ca", "cp", "cv", "npmi", "uci", and "umass".

palmetto = Palmetto()

print('ARX LDA Okapi LV')

topics =  [['flow', 'simulation', 'velocity', 'fluid', 'dynamic', 'numerical', 'force', 'particle', 'model', 'plasma'],
['collision', 'production', 'section', '0', 'cross', 'energy', 'decay', 'gev', 'experiment', 'proton'],
['network', 'model', 'social', 'biological', 'individual', 'population', 'dynamic', 'human', 'cell', 'activity'],
['algorithm', 'method', 'propose', 'learn', 'performance', 'task', 'image', 'art', 'network', 'base'],
['equation', 'function', 'solution', 'operator', 'matrix', 'prove', 'obtain', 'integral', 'space', 'condition'],
['system', 'master', 'bath', 'markovian', 'equilibrium', 'damp', 'heat', 'thermodynamics', 'dynamic', 'steady'],
['group', 'algebra', 'manifold', 'prove', 'space', 'give', 'mathbb', 'show', 'theorem', 'invariant'],
['gauge', 'higgs', 'theory', 'loop', 'symmetry', 'model', 'mass', 'su', 'scalar', 'standard'],
['graph', 'bound', 'problem', 'algorithm', 'number', 'set', 'prove', '1', 'vertex', 'complexity'],
['al', 'phys', '2015', 'arxiv', 'rev', '2014', '2013', 'bf', '2012', '2016'],
 ['star', 'galaxy', 'ray', 'observation', 'mass', '10', '0', 'stellar', 'emission', '1'],
 ['gravity', 'black', 'field', 'hole', 'scalar', 'gravitational', 'cosmological', 'einstein', 'theory', 'universe'],
 ['material', 'surface', 'electronic', 'electron', 'device', 'crystal', 'temperature', 'layer', 'metal', 'film'],
 ['channel', 'information', 'communication', 'quantum', 'capacity', 'network', 'protocol', 'system', 'scheme', 'state'],
 ['optical', 'photon', 'laser', 'frequency', 'atom', 'pulse', 'light', 'quantum', 'mode', 'cavity'],
 ['model', 'data', 'method', 'estimation', 'distribution', 'propose', 'sample', 'estimate', 'problem', 'optimal'],
 ['calculation', 'quark', 'state', 'nuclear', 'qcd', 'meson', 'calculate', 'nucleon', 'nucleus', 'neutron'],
 ['phase', 'spin', 'state', 'transition', 'temperature', 'system', 'lattice', 'quantum', 'interaction', 'topological'],
 ['review', 'physic', 'development', 'discuss', 'recent', 'research', 'year', 'science', 'concept', 'understanding']]

cv =[]
for words in topics:
    print(words, ':', round(palmetto.get_coherence(words),2))
    cv.append(palmetto.get_coherence(words))

print(round(np.mean(cv), 2), "+/-", round(np.std(cv), 2))

print('\n\nARX LDA Okapi CVV')
topics = [['flow', 'velocity', 'fluid', 'surface', 'simulation', 'force', 'hydrodynamic', 'numerical', 'motion', 'rotation'],
['quark', 'collision', 'production', 'section', 'decay', 'cross', 'heavy', 'mass', 'data', 'gev'],
['network', 'dynamic', 'cell', 'model', 'population', 'biological', 'protein', 'process', 'mechanism', 'individual'],
['algorithm', 'performance', 'problem', 'method', 'estimation', 'paper', 'optimal', 'error', 'channel', 'rate'],
['equation', 'system', 'solution', 'function', 'time', 'numerical', 'dynamic', 'model', 'state', 'method'],
['magnetic', 'optical', 'spin', 'material', 'temperature', 'electron', 'band', 'crystal', 'phase', 'field'],
['formal', 'theory', 'notion', 'logic', 'language', 'proof', 'category', 'paper', 'automaton', 'calculus'],
['group', 'algebra', 'manifold', 'space', 'mathbb', 'class', 'theorem', 'lie', 'paper', 'algebraic'],
['research', 'year', 'social', 'development', 'market', 'paper', 'price', 'science', 'physic', 'decade'],
['graph', 'bound', 'random', 'number', 'problem', 'algorithm', 'set', 'vertex', 'log', 'edge'],
 ['atom', 'interaction', 'state', 'atomic', 'excitation', 'quantum', 'energy', 'effect', 'transition', 'dipole'],
 ['network', 'method', 'data', 'task', 'image', 'neural', 'learning', 'art', 'approach', 'feature'],
 ['gravity', 'black', 'hole', 'field', 'spacetime', 'equation', 'gravitational', 'theory', 'einstein', 'solution'],
 ['star', 'galaxy', 'observation', 'stellar', 'mass', 'emission', 'ray', 'survey', 'galactic', 'source'],
 ['van', 'neumann', 'von', 'recurrence', 'der', 'periodicity', 'waals', 'corner', 'cpt', 'rough'],
 ['plasma', 'energy', 'ray', 'electron', 'solar', 'high', 'ion', 'radiation', 'beam', 'detector'],
 ['dark', 'matter', 'model', 'higgs', 'standard', 'inflation', 'mass', 'scale', 'scalar', 'scenario'],
 ['theory', 'gauge', 'loop', 'symmetry', 'field', 'renormalization', 'dimension', 'operator', 'chiral', 'dimensional'],
 ['quantum', 'state', 'information', 'system', 'measurement', 'entanglement', 'qubit', 'operation', 'technology', 'protocol']]


cv =[]
for words in topics:
    print(words, ':', round(palmetto.get_coherence(words),2))
    cv.append(palmetto.get_coherence(words))

print(round(np.mean(cv), 2), "+/-", round(np.std(cv), 2))