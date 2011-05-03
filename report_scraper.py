from BeautifulSoup import BeautifulSoup
from BeautifulSoup import BeautifulStoneSoup
import urllib2
import re
import os

def fetch(url):
    opener = urllib2.build_opener()
    opener.addheaders = [("User-agent", "report-crawler-bot")]
    response = opener.open(url)
    return response.read()

def store_report(report_url, save_dir):
    name = report_url.rpartition('/')[2]
    name = name.replace('.asp','.html')
    print 'Fetching',name
    doc = fetch(report_url).strip()
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

def crawl_reports(year):
    output_dir = '../data/rir/reports/'+str(year)
    base_url = 'http://www.tsb.gc.ca/eng/rapports-reports/rail/'+year+'/'
    index = fetch(base_url).strip()
    report_urls = extract_report_urls(index)
    report_urls = [base_url+rel for rel in report_urls]
    for url in report_urls:
        store_report(url, output_dir)

if __name__=='__main__':
    cats = ['precedentes-earlier','etudes-studies']
    cats += [str(y) for y in range(1994,2011)]
    for c in cats:
        crawl_reports(c)
