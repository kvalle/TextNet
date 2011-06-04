"""Utility functions facititating easy plotting with matplotlib.

Functions of note:

- :func:`plot`: plot a regular plot, given input x,y-coordinates.
- :func:`bar_graph`: plot a horizontal bar graph from x-coordinates and named
  groups of lists of y-corrdinates.
- :func:`histogram`: plots a histogram from a set of samples and a given numbe of bins.
- :func:`plot_degree_distribution`: plot the degree distribution provided a networkx graph.

:Author: Kjetil Valle <kjetilva@stud.ntnu.no>"""

import matplotlib.pyplot as plt
from nltk.probability import FreqDist
import networkx as nx
import numpy as np

def plot(x_vals, y_vals, x_label, y_label, title, axis=None, legend_place='lower right'):
    """Plot a 2D plot from provided x- and y-values.

    Given lists of *x_vals* and *y_vals*, a plot is drawn. Axes are
    labeled using *x_label* and *y_label*.

    Plot can be cropped/scaled with *axis* as a 4-tuple of (x_min,x_max,y_min,y_max).

    Place legend with *legend_place* as string argument matching
    /(lower|middle|upper) (right|center|left)/.
    """
    y_max = None
    y_min = None

    for name, vals in y_vals.items():
        plt.plot(x_vals, vals, linewidth=2.0, label=name)
        if  y_min is None or min(vals) < y_min:
            y_min = min(vals)
        if  y_max is None or max(vals) > y_max:
            y_max = max(vals)

    x_min = min(x_vals)
    x_max = max(x_vals)
    if axis is None:
        plt.axis([x_min,x_max,y_min,y_max])
    else:
        plt.axis(axis)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.title(title)
    if legend_place: plt.legend(loc=legend_place)
    plt.show()

def plot_degree_distribution(g):
    """Plot degree distribution for a graph *g*"""
    degree_sequence=sorted(nx.degree(g).values(),reverse=True) # degree sequence
    dmax=max(degree_sequence)

    plt.loglog(degree_sequence,'b-',marker='.')
    plt.title("Degree distribution")
    plt.ylabel("degree")
    plt.xlabel("rank")

    ### draw graph in inset
    #~ plt.axes([0.45,0.45,0.45,0.45])
    #~ Gcc=nx.connected_component_subgraphs(g)[0]
    #~ pos=nx.spring_layout(Gcc)
    #~ plt.axis('off')
    #~ nx.draw_networkx_nodes(Gcc,pos,node_size=20)
    #~ nx.draw_networkx_edges(Gcc,pos,alpha=0.4)

    #~ plt.savefig("degree_histogram.png")
    plt.show()


def histogram(samples, x_label='', y_label='', title='', axis=None, bins=10, range=None):
    """Create histogram from a set of *samples*.

    Number of *bins* and *range* can be specified in the same manner as to
    :func:`numpy.histogram`.

    Crop/scale with *axis* as a 4-tuple of (x_min,x_max,y_min,y_max).
    """
    fig = plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    ax = fig.add_subplot(111)

    samples = np.array(samples)
    hist, bin_edges = np.histogram(samples,bins=bins,range=range)
    ax.hist(samples, bin_edges, rwidth=1.0)

    plt.show()
    return hist, bin_edges


def bar_graph(data, bar_names, x_label='', y_label='', title='', axis=None, colors=None, legend_place='lower right'):
    """Create horzontal bar chart with lists of data values.

    Plots a bar chart given a dictionary of *data* with a type as key, and a sequence of
    values corresponding to elements in *bar_names* as value.

    Place legend with *legend_place* as string argument matching
    /(lower|middle|upper) (right|center|left)/.
    """
    from matplotlib import cm
    fig = plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    ax = fig.add_subplot(111)

    num_groups = len(data.values()[0])
    group_size = len(data.values())
    yvals = np.arange(num_groups)
    width= 0.8/len(data.values())

    ps = []
    for i, vals in enumerate(data.values()):
        if colors is None:
            color = cm.spectral(1.*i/group_size) # colormaps: gist_rainbow, jet, hsv, spectral, ..
        else:
            color = colors[i%len(colors)]
        p = ax.barh(yvals+(width*i), vals, width, color=color)
        ps.append(p[0])

    plt.yticks(yvals+width, bar_names)
    if legend_place is not None:
        plt.legend( ps, data.keys(), loc=legend_place)

    plt.show()

def demo():
    x_vals = range(0,352,10)
    y_vals = {
                'MaxAvg_old': [0.0, 0.683, 0.866, 0.921, 0.909, 0.913, 0.906, 0.904, 0.908, 0.912, 0.910, 0.914, 0.908, 0.912, 0.913, 0.914, 0.919, 0.921, 0.913, 0.920, 0.924, 0.921, 0.915, 0.921, 0.918, 0.920, 0.919, 0.921, 0.919, 0.918, 0.916, 0.922, 0.910, 0.914, 0.915, 0.912],
                'MaxAvg_new': [0.0, 0.68175000000000063, 0.86125000000000029, 0.92375000000000029, 0.90841666666666587, 0.91383333333333272, 0.90666666666666607, 0.90833333333333266, 0.91125000000000034, 0.91191666666666593, 0.91316666666666557, 0.91233333333333277, 0.90649999999999886, 0.90458333333333318, 0.91058333333333341, 0.91416666666666602, 0.91616666666666602, 0.91766666666666641, 0.91366666666666552, 0.91483333333333261, 0.92175000000000051, 0.91525000000000056, 0.91641666666666688, 0.92258333333333231, 0.91725000000000034, 0.92208333333333314, 0.92049999999999987, 0.92158333333333242, 0.91733333333333222, 0.91833333333333267, 0.91933333333333311, 0.91558333333333342, 0.9121666666666669, 0.91516666666666646, 0.91899999999999982, 0.91474999999999917],
                'Random_old': [0.0, 0.406, 0.450, 0.503, 0.549, 0.767, 0.830, 0.840, 0.783, 0.857, 0.776, 0.756, 0.814, 0.836, 0.850, 0.832, 0.853, 0.802, 0.840, 0.864, 0.853, 0.891, 0.919, 0.918, 0.885, 0.886, 0.883, 0.918, 0.855, 0.906, 0.866, 0.926, 0.921, 0.917, 0.917, 0.917],
                'Random_new': [0.0, 0.38441666666666657, 0.54891666666666694, 0.59566666666666734, 0.75758333333333405, 0.55624999999999969, 0.83941666666666626, 0.76166666666666594, 0.73074999999999957, 0.87558333333333282, 0.85224999999999906, 0.7864166666666671, 0.77066666666666706, 0.87866666666666782, 0.82291666666666674, 0.83883333333333376, 0.89199999999999979, 0.85849999999999915, 0.84016666666666717, 0.89000000000000057, 0.90366666666666673, 0.89441666666666697, 0.9059166666666667, 0.92875000000000019, 0.86266666666666691, 0.88908333333333323, 0.91899999999999993, 0.88816666666666666, 0.89133333333333398, 0.88300000000000056, 0.9066666666666664, 0.93483333333333285, 0.91699999999999959, 0.89008333333333367, 0.9176666666666663, 0.91416666666666568],
                'Shapley_old': [0.0, 0.468, 0.690, 0.833, 0.854, 0.862, 0.856, 0.859, 0.870, 0.875, 0.873, 0.866, 0.867, 0.860, 0.893, 0.898, 0.899, 0.898, 0.901, 0.908, 0.910, 0.896, 0.904, 0.901, 0.909, 0.902, 0.900, 0.904, 0.913, 0.917, 0.919, 0.919, 0.921, 0.925, 0.924, 0.914],
                'Shapley_new': [0.0, 0.87208333333333321, 0.89133333333333287, 0.90649999999999931, 0.90033333333333332, 0.90358333333333307, 0.89174999999999904, 0.88308333333333289, 0.91516666666666691, 0.91333333333333244, 0.91741666666666677, 0.90683333333333294, 0.90166666666666651, 0.90783333333333371, 0.91999999999999915, 0.90849999999999942, 0.91574999999999995, 0.92216666666666625, 0.91683333333333272, 0.92408333333333281, 0.91858333333333375, 0.93466666666666565, 0.9346666666666662, 0.92666666666666631, 0.93008333333333371, 0.92775000000000019, 0.93433333333333235, 0.92583333333333295, 0.93674999999999964, 0.92274999999999929, 0.9242499999999999, 0.93274999999999919, 0.92766666666666686, 0.92524999999999935, 0.92216666666666569, 0.91400000000000003]
                }
    x_label = 'Average accuracy'
    y_label = 'Number of features'
    title = 'Accuracy of feature selectors'
    plot(x_vals, y_vals, x_label, y_label, title, [0,x_vals[-1],0,1])

def test_plot_degree_distribution():
    #~ g=nx.erdos_renyi_graph(100,0.15)
    #~ g=nx.watts_strogatz_graph(1000,3,0.1)
    g = nx.barabasi_albert_graph(100,2,)
    plot_degree_distribution(g)

def test_histogram():
    #~ histogram([1,2,3,4,3,2,3,4,3,3,1,3,4],'x-axis','y-axis','histogram test')
    bar_names = ['PageRank','Degree','Closeness','Betweenness','Current-flow Closeness','PageRank','Degree','Closeness','Betweenness','Current-flow Closeness','Betweenness','Current-flow Closeness']
    data = {'weighted':np.array([3,4,1,5,2,3,4,1,5,2,1,4]),
        #~ 'unweighted':np.array([4,5,1,3,6,4,5,1,3,6,4,6]),
        'foo bar':np.array([4,5,1,3,6,4,5,1,3,6,4,6])}
    colors = ['#3C54FF','#EF4C32','#27A713']
    bar_graph(data, bar_names, colors=None)

def tikz_barchart(data, labels, scale = 1.0, yscale=1.0, xscale=1.0, bar_widths = [7,5], base=.3, skip=0.4, legend=None, grid=True, pad=True, color='darkgray', labels_only=False, axsis=True, legend_sep=1.0, scale_height=True, low_cut=0.0, y_tics=None, tick=True):
    def data_to_coords(data, base, skip):
        res = []
        for i in range(len(data[0])):
            b = base
            s = ''
            for d in data:
                s += '('+str(b)+','+str(d[i])+') '
                b += skip
            res.append(s)
        return res

    def print_labels(labels, base, skip):
        res = ''
        b = base + 0.1
        for name in labels:
            res += '\draw ('+str(b)+','+str(-lh/2)+') node[left, rotate=30] {\small '+name+'};\n'
            b += skip
        return res

    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] -= low_cut
    if scale_height:
        max_val = max(max(data))
        max_val += 0.1 - max_val % 0.1
    else:
        max_val = 1.0
    lh = 0.16/yscale
    bars = data_to_coords(data, base, skip)
    width = len(data)*skip + float(base)/2
    if pad:
        width += 0.2 - width % 0.2
    else:
        width -= width%0.2

    fig =  '\\begin{tikzpicture}[ybar,scale='+str(scale)+',xscale='+str(xscale)+',yscale='+str(yscale)+']\n'

    if labels:
        fig += '% Labels\n'
        fig += print_labels(labels, base, skip)

    if grid and not labels_only:
        fig += '% Grid\n'
        fig += '\draw[help lines,ystep=0.1cm,xstep='+str(width)+'] (0,0) grid ('+str(width)+','+str(max_val)+');\n\n'

    if legend and not labels_only:
        legend_start = 0.2
        fig += '% Legend\n'
        fig += '\draw[color='+color+',fill='+color+'!80] ('+str(legend_start)+','+str(max_val+lh)+') rectangle ('+str(legend_start+0.2)+','+str(max_val+2*lh)+');\n'
        fig += '\draw ('+str(legend_start+0.2)+','+str(max_val+lh+lh/2)+') node[right] {\small '+legend[0]+'};\n'
        fig += '\draw[color='+color+'!50,fill='+color+'!20] ('+str(legend_start+legend_sep)+','+str(max_val+lh)+') rectangle ('+str(legend_start+legend_sep+0.2)+','+str(max_val+2*lh)+'); node[right] {\small Weighted};\n'
        fig += '\draw ('+str(legend_start+legend_sep+0.2)+','+str(max_val+lh+lh/2)+') node[right] {\small '+legend[1]+'};\n\n'

    if not labels_only:
        fig += '% Bars\n'
        fig += '\draw[color='+color+',fill='+color+'!80,bar width='+str(bar_widths[0])+']\n'
        fig += '    plot coordinates{'+bars[0]+'};\n'
        fig += '\draw[color='+color+'!50,fill='+color+'!20,bar width='+str(bar_widths[1])+',bar shift='+str(bar_widths[0]/3)+', thick] \n'
        fig += '    plot coordinates{'+bars[1]+'};\n\n'

    if axsis and not labels_only:
        fig += '% Y-axis\n'
        fig += '\draw[->, thick] (0,0) -- (0,'+str(max_val)+');\n'
        fig += '\draw[-, thick] (0,0) -- ('+str(width)+',0);\n'
        fig += '\\foreach \y/\ytext in {'
        if not y_tics:
            y_tics = {0:'0\%', .1:'', .2:'20\%', .3:'', .4:'40\%', .5:'', .6:'60\%', .7:'', .8:'80\%', .9:'', 1:'100\%'}
        ys = []
        for i in y_tics:
            y = i-low_cut
            if y>=0.0:
                ys.append([y,y_tics[i]])
        fig += ','.join([str(y[0])+'/'+y[1] for y in ys if y[0] <= max_val])
        #~ '.1/,.2/20\%,.3/,.4/40\%,.5/,.6/60\%,.7/,.8/80\%,.9/,1/100\%
        fig += '}\n'
        if tick:
            fig += '\draw[yshift=\y cm] (1pt,0pt) -- (-1pt,0pt) node[left,fill=white] {$\ytext$};\n\n'
        else:
            fig += '\draw[yshift=\y cm] (-1pt,0pt) node[left,fill=white] {$\ytext$};\n\n'

    fig += '\end{tikzpicture}\n'
    return fig

if  __name__=='__main__':
    #~ plot_context_sizes()
    test_histogram()
    #~ test_plot_degree_distribution()
