"""Module for reading and writing case files.

The read_* methods provide reading of cases in various formats from dataset.
For converting dataset between formats, use the appropriate create_dataset_* function.
It is also possible to provide custom conversion functions to the :func:`create_dataset` function.


The following formats are supported for dataset conversion:

:HTML:
  Expected formatted similarly to AIR dataset reports for conversion to cases.
  Conversion to text/dependencies should work regardless.
:Text:
  Raw text. Anything within p if extracted from HTML.
:Preprocessed text:
  Processed using the default parameters from preprocess.preprocess_text().
:Dependencies:
  As defined by the stanford dependency parser.

The module expects to work with datasets structured so that each category
is in a separate subfolder named after the category.

:Author: Kjetil Valle <kjetilva@stud.ntnu.no>"""

import os
import os.path
import re
import tempfile
import shutil
import pickle
from HTMLParser import HTMLParseError

import preprocess
import report_data

######
##
##  Reading files
##
######

file_names = {}

def get_file_names(in_path):
    """Retrieve list of file/case names matching dataset path."""
    return file_names[in_path]

def read_files(in_path, unpickle_content=False, verbose=False):
    """Read dataset/cases from files.

    Files are read recursively from *in_path*, with subdirectory names
    used as labels."""
    if in_path not in file_names:
        file_names[in_path] = []
    docs =  []
    labels = []
    for class_name in os.listdir(in_path):
        class_path = os.path.join(in_path, class_name)
        for doc_name in os.listdir(class_path):
            doc_path = os.path.join(class_path, doc_name)
            file_names[in_path].append(doc_name)
            doc_text = read_file(doc_path, verbose)
            if unpickle_content:
                doc_text = pickle.loads(doc_text)
            docs.append(doc_text)
            labels.append(class_name)
    return (docs, labels)

def read_file(file_path, verbose=False):
    """Fetch contents of a single file from *file_path*."""
    if verbose: print '   ', file_path
    with open(file_path, 'r') as f:
        words = f.read()
    return words

def read_data(path, graph_types):
    #TODO: refactor this ugly function!
    preprocessed = None
    d = {}
    for gtype in graph_types:
        if gtype=='co-occurrence':
            if preprocessed == None:
                preprocessed = read_files(path+'_preprocessed')
            d[gtype] = preprocessed
        if gtype=='random':
            if preprocessed == None:
                preprocessed = read_files(path+'_preprocessed')
            d[gtype] = preprocessed
        if gtype=='dependency':
            d[gtype] = read_files(path+'_dependencies')
    return d

######
##
##  Writing files
##
##  Various kinds of preprocessing of data:
##  raw -> text
##      -> text_processed
##      -> dependencies
##
##  In the case where raw is text only, the two will be equal, when raw
##  data is in html format or similar, text is extracted.
##  Dependencies are crated using the stanford dependency parser, and
##  subsequently serialized to file using pickle format.
##
######

def _html_to_text(raw_html):
    """Convert HTML to text"""
    parser = report_data.ReportParser('', raw=raw_html)
    parser.close()
    return parser.report.str()

def _text_to_preprocessed_text(text):
    """Convert text to preprocessed text"""
    prep = preprocess.preprocess_text(text)
    return ' '.join(prep)

def _text_to_dependencies(text):
    """Covert text to set of stanford dependencies"""
    deps = preprocess.extract_dependencies(text)
    return pickle.dumps(deps)

def _html_to_problem_description(raw_html):
    """Convert HTML to problem description text.

    Problem description parts of HTML report are extracted and returned
    as raw text."""
    parser = report_data.ReportParser('', raw=raw_html)
    parser.close()
    case = report_data.ReportCase(parser.report)
    return case.description

def _html_to_solution(raw_html):
    """Convert HTML to problem solution text.

    Problem solution parts of HTML report are extracted and returned
    as raw text."""
    parser = report_data.ReportParser('', raw=raw_html)
    parser.close()
    case = report_data.ReportCase(parser.report)
    return case.solution

def create_dataset_html_to_text(base_path, target_path):
    """Convert dataset: HTML to text"""
    create_dataset(base_path, target_path, _html_to_text)

def create_dataset_html_to_case(base_path, target_path):
    """Convert dataset: HTML to CBR case"""
    create_dataset(base_path, target_path+"_problem_descriptions", _html_to_problem_description)
    create_dataset(base_path, target_path+"_solutions", _html_to_solution)

def create_dataset_text_to_preprocessed_text(base_path, target_path):
    """Convert dataset: Text to preprocessed text"""
    create_dataset(base_path, target_path, _text_to_preprocessed_text)

def create_dataset_text_to_dependencies(base_path, target_path):
    """Convert dataset: Text to stanford dependencies"""
    create_dataset(base_path, target_path, _text_to_dependencies)

def create_dataset(base_path, target_path, processing_fn):
    """
    Crate a new dataset in target_path from data in base_path.
    Every file in base_path is processed using function processing_fn,
    and then stored under target_path.

    *base_path* is the path to the data to be processed and turned into new dataset.
    *target_path* is the name/path of the new dataset.
    The *processing_fn* function is used for processing each document.
    *processing_fn* needs to have string as both input and output.
    """
    if base_path==target_path: # check that we are not trying to read and write from same folder
        raise Exception('base and target paths cannot be the same')

    if not os.path.exists(base_path): # check that base path exists
        raise Exception('base path does not exist')

    if os.path.exists(target_path): # confirm overwriting existing folders
        inp = raw_input('> Directory %s already exist!\n> Continue? (Y/n): ' % target_path)
        if inp.lower().find('y') == -1:
            print '> Aborted.'
            return

    overwrite_files = None
    for dir_path, dir_names, file_names in os.walk(base_path):
        # create new directories in target for each in base
        for dir_name in dir_names:
            class_dir = os.path.join(dir_path,dir_name)
            class_dir = re.sub(base_path, target_path, class_dir)
            try:
                os.makedirs(class_dir)
                print '> Created directory %s' % class_dir
            except OSError as e:
                print '> Directory %s already existed' % class_dir
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            new_path = re.sub(base_path, target_path, file_path)
            if os.path.exists(new_path): # confirm overwriting existing files
                if overwrite_files is None:
                    inp = raw_input('> File %s already exist!\n> Overwrite existing files? (y/N): ' % new_path)
                    if inp.lower().find('y') == -1:
                        overwrite_files = False
                    else:
                        overwrite_files = True
                if not overwrite_files:
                    print '> Skipped existing file %s' % new_path
                    continue
            print '> Processing %s' % file_path
            try:
                # open files (original and tmp)
                fh, tmp_path = tempfile.mkstemp()
                new_file = open(tmp_path, 'w+')
                old_file = open(file_path)

                # do preprocessing
                text = old_file.read()
                text = processing_fn(text)
                new_file.write(text)

                # clean up and create new permanent file
                new_file.close()
                os.close(fh)
                old_file.close()
                shutil.move(tmp_path, new_path)
            except HTMLParseError as error:
                print error
            finally:
                if not new_file.closed: new_file.close()
                if not old_file.closed: old_file.close()

def write_to_file(data, filename):
    """Dump data to file"""
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(filename, 'a+') as f:
        f.write(data)

######
##
##  Pickle methods
##
######

def pickle_from_file(filename):
    """Read file and unpickle contents"""
    try:
        pkl_file = open(filename, 'rb')
    except IOError as e:
        print '! Unable to open:', filename
        return None
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data

def pickle_to_file(data, filename):
    """Pickle contents and dump to file"""
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    output = open(filename, 'a+')
    pickle.dump(data, output)
    output.close()

###### encoding utility functions

def test_ascii(path='../data/air/reports_text'):
    """Test whether documents in dataset are ascii encoded"""
    (documents, labels) = read_files(path)
    names = get_file_names(path)
    for i, doc in enumerate(documents):
        try:
            doc = doc.encode('ascii')
        except UnicodeDecodeError as e:
            print names[i]
    print "done"

def fix_ascii(path):
    """Test whether documents in dataset are ascii encoded"""
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        for doc_name in os.listdir(class_path):
            doc_path = os.path.join(class_path, doc_name)
            with open(doc_path, 'r') as f:
                doc = f.read()
            doc = unicode(doc, encoding='ascii', errors='ignore')
            with open(doc_path, 'w') as f:
                f.write(doc)
    print "done"

if __name__ == "__main__":
    #~ test_ascii()
    #~ create_dataset_html_to_text('../data/air/html', '../data/air/text')
    #~ create_dataset_html_to_case('../data/air/html', '../data/air/text')
    #~ create_dataset_text_to_preprocessed_text('../data/air/problem_descriptions_text', '../data/air/problem_descriptions_preprocessed')
    #~ create_dataset_text_to_dependencies('../data/tasa/TASA900_text', '../data/tasa/TASA900_dependencies_2')
    #~ create_dataset_text_to_dependencies('../data/air/problem_descriptions_text', '../data/air/problem_descriptions_dependencies_2')

    #~ print pickle_from_file('../data/air/problem_descriptions_dependencies/1999/a99o0244.html')
    #~ create_dataset_text_to_preprocessed_text('../data/tasa/TASATest2/test_text', '../data/tasa/TASATest2/test_preprocessed')

    #~ create_dataset_text_to_preprocessed_text('../data/mir/problem_descriptions_text', '../data/mir/problem_descriptions_preprocessed')
    #~ create_dataset_text_to_preprocessed_text('../data/mir/solutions_text', '../data/mir/solutions_preprocessed')
    #~ create_dataset_text_to_dependencies('../data/mir/problem_descriptions_text', '../data/mir/problem_descriptions_dependencies')
    create_dataset_text_to_dependencies('../data/mir/solutions_text', '../data/mir/solutions_dependencies')
