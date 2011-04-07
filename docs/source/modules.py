modules = ['data',
            'report_data',
            'preprocess',
            'stanford_parser',
            'freq_representation',
            'graph_representation',
            'graph',
            'classify',
            'retrieval',
            'evaluation',
            'plotter',
            'util']

for m in modules:
    print
    print m
    print '-'*len(m)
    print
    print '.. automodule:: '+m
    print '.. toctree::'
    print
    print '   Read more <'+m+'>'
    print
