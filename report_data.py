"""
Helper module for 'data', used to extract problem description and solution parts from cases.

Parses reports formatted in HTML, structured as those in the AIR dataset, and split them into
problem description part and solution part of textual CBR cases.
Solutions are identified based on section titles in the reports.
Titles matching words such as 'finding' or 'conclusion' are considered as part of the solution.
The remaining report is by default the problem description.

@author: Gleb Sizov <sizov@idi.ntnu.no>
"""

from HTMLParser import HTMLParser, HTMLParseError
import re
import os

class Section(object):
    """
    Section brunch (tree structure) of a report.
    On the top level represents the report itself.

    Contains the following information:
    - report title: string
    - section level: number
    - contained (sub)sections: list of Section instances
    - section paragraphs: list of strings
    - meta information: list of strings
    """

    def __init__(self, title = '', parent = None):
        self.title = title
        self.level = 0
        self.sections = []
        self.paragraphs = []
        self.meta = []

        self.parent = parent
        if parent:
            parent.sections.append(self)

        self.root = self.get_root()
        self.level = self.get_level()

    def titles(self):
        titles = [self.title]
        for s in self.sections:
            titles.append(s.title)
        return titles

    def str(self, sep = ' '):
        return sep.join([self.title,
                         sep.join(self.paragraphs),
                         sep.join([str(s) for s in self.sections])])

    def __str__(self):
        return self.str(' ')

    def get_root(self):
        return self.parent.get_root() if self.parent else self

    def get_level(self):
        return self.parent.get_level() + 1 if self.parent else 0

    def accept(self, visitor):
        #implements visitor pattern
        # visit() may return some value different from None to avoid visiting subsections
        if not visitor.visit(self):
            for s in self.sections:
                s.accept(visitor)

class ReportParser(HTMLParser):
    """
    Parser for canadian html reports. It's quite a hack so be carefull messing with it.
    """

    def __init__(self, path, raw=None):
        HTMLParser.__init__(self)
        self.report = Section()

        self.data_tags = set(['title', 'p', 'ol', 'ul'])
        self.to_buffer = False
        self.to_report = False
        self.to_meta = False

        self.buffer = ''
        self.sections = [None] * 6
        self.sections[0] = self.report

        if raw is None:
            with open(path, 'r') as f:
                raw = f.read()

        self.h_re = re.compile('h([1-6])', re.IGNORECASE)
        self.sub_re = '&.{2,8};|\n|<br */>|<img[^>]*/>'
        self.feed(raw)

    def handle_starttag(self, tag, attrs):
        if attrs and attrs[0] == ('class', 'reportInfo'):
            self.to_meta = True
            self.buffer = ''

        if self.to_meta:
            self.to_buffer = True
        elif self.h_re.match(tag) or tag in self.data_tags:
            self.to_buffer = True
            self.buffer = ''

    def handle_endtag(self, tag):
        if tag == 'title':
            self.report.title = self.buffer.strip()
        elif self.to_meta:
            if tag == 'div':
                self.to_meta = False
            elif tag in ['p', 'br']:
                meta = self.buffer.strip()
                if meta and 'report' not in meta.lower():
                    self.report.meta.append(meta)
                self.buffer = ''

        res = self.h_re.match(tag)
        tmp = self.buffer.lower()

        if res and ('summary' in tmp or 'synopsis' in tmp or 'factual information' in tmp): #ignore text before these sections.
            self.to_report = True
        elif "report concludes the transportation safety board" in tmp or 'table of contents' in tmp: #ignore text after these sections.
            self.to_report = False

        if self.to_report:
            self.buffer = self.buffer.strip()

            if res:
                level = int(res.groups()[0]) - 1
                while not self.sections[level - 1]:
                    level -= 1

                self.section = self.sections[level] = Section(self.buffer, self.sections[level - 1])
            elif tag in self.data_tags:
                if len(self.buffer) > 30:
                    self.section.paragraphs.append(' '.join(re.sub(self.sub_re, ' ', self.buffer).split()))

    def handle_data(self, data):
        if self.to_buffer:
            self.buffer += data

class Case(object):
    def __init__(self, description = '', solution = ''):
        self.description = description
        self.solution = solution

class ReportCase(Case):
    """
    Splits report into description and solution parts based on the section titles.
    """

    #~ solution_titles = ["analysis", "finding", "causes", "contributing factors", "safety action", "conclusion"]
    solution_pattern = '.*(finding|conclusion|safety|analysis|causes).*'

    def __init__(self, report):
        super(ReportCase, self).__init__()
        report.accept(self)

    def visit(self, section):
        #~ if sum([t in section.title.lower() for t in self.solution_titles]):
        if re.search(self.solution_pattern, section.title.lower()) is None:
            #~ self.description += ' '.join(section.paragraphs)
            self.description += str(section)
            return True
        elif section.parent:
            self.solution += ' '.join(section.paragraphs)
            self.solution += str(section)
            return True

    def __str__(self):
        return '\n' + '\n'.join([self.description, self.solution])

def load_reports(path):
    reports = []

    for dir_path, dir_names, file_names in os.walk(path):
        for file_name in file_names:
            try:
                reports.append(load_report(os.path.join(dir_path, file_name)))
            except HTMLParseError as error:
                print error

    print str(len(reports)) + ' reports loaded'
    return reports

def load_report(path):
    print 'loading ' + path
    parser = ReportParser(path)
    parser.close()
    return parser.report

def test():
    causes = {}
    report_length = {}
    num_reports = 0
    import preprocess
    solution_pattern = '.*(finding|conclusion|safety|analysis|causes).*'
    for r in load_reports('../data/air/html'):
        titles = r.titles()
        s = []
        pd = []
        for t in titles:
            match = re.search(solution_pattern, t.lower())
            if match is None:
                pd.append(t)
            else:
                s.append(t)
        print len(titles),'---',len(s),'---',len(pd)

def test_case(report=None):
    if report==None:
        report = load_report('../data/air/html2/test/a04h0001.html')
    case = ReportCase(report)
    for section in report.sections:
        print section.title
        for subsec in section.sections:
            print '   ',subsec.title
    print
    #~ print case.description
    #~ print
    #~ print case.solution
    print
    import preprocess
    print len(preprocess.tokenize_tokens(case.solution))
    print len(preprocess.tokenize_tokens(case.description))

if __name__ == "__main__":
    #~ test()
    test_case()
