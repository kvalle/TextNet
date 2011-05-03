from BeautifulSoup import BeautifulSoup
from BeautifulSoup import BeautifulStoneSoup
import urllib2
import re
import os

def fetch(url):
    opener = urllib2.build_opener()
    opener.addheaders = [("User-agent", "report-crawler")]
    response = opener.open(url)
    return response.read()

def fetch_extended_report(report_url):
    alt_url = report_url.replace('.asp','_index.asp')
    try:
        doc = fetch(alt_url).strip()
    except urllib2.HTTPError as e:
        # not a problem, just no alternate for this report
        return False
    sec_num = 1
    while True:
        alt_url = report_url.replace('.asp','_sec'+str(sec_num)+'.asp')
        try:
            sec = fetch(alt_url).strip()
            doc += '\n\n'+sec
        except urllib2.HTTPError as e:
            # 404, no more sections
            break
        sec_num += 1
    return doc

def store_report(report_url, save_dir):
    alt_url = report_url.replace('.asp','_index.asp')
    name = report_url.rpartition('/')[2]
    name = name.replace('.asp','.html')
    print 'Fetching',name
    doc = fetch(report_url).strip()
    alt = fetch_extended_report(report_url)
    if alt:
        doc += '\n\n'+alt
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = save_dir+'/'+name
    with open(filename, 'w') as f:
        f.write(doc)

def extract_report_urls(index):
    soup = BeautifulSoup(index)
    atags = soup.findAll('a')
    urls = []
    for a in atags:
        if a.string=='HTML':
            urls.append(a['href'])
    return urls

def crawl_reports(url, year, output_dir):
    base_url = url+year+'/'
    index = fetch(base_url).strip()
    report_urls = extract_report_urls(index)
    report_urls = [base_url+rel for rel in report_urls]
    for r_url in report_urls:
        store_report(r_url, output_dir+'/'+year)

def fetch_rail():
    cats = ['precedentes-earlier','etudes-studies']
    cats += [str(y) for y in range(1994,2011)]
    base_url = 'http://www.tsb.gc.ca/eng/rapports-reports/rail/'
    data_dir = '../data/rir/reports/'
    for c in cats:
        crawl_reports(base_url, c, data_dir)

def fetch_marine():
    cats = ['precedentes-earlier','etudes-studies']
    cats += [str(y) for y in range(1994,2010)]
    base_url = 'http://www.tsb.gc.ca/eng/rapports-reports/marine/'
    data_dir = '../data/mir/reports/'
    for c in cats:
        crawl_reports(base_url, c, data_dir)

def fetch_pipeline():
    cats = ['1994','1995','1996','1997','1999','2000','2001','2002','2005','2006', '2007','2009']
    base_url = 'http://www.tsb.gc.ca/eng/rapports-reports/pipeline/'
    data_dir = '../data/pir/reports/'
    for c in cats:
        crawl_reports(base_url, c, data_dir)

if __name__=='__main__':
    fetch_marine()
    fetch_rail()
    fetch_pipeline()
