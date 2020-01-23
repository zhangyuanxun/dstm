from bs4 import BeautifulSoup
import urllib2
import json
import sys
from contants import *


def collector():
    base_url = 'https://journals.plos.org/ploscompbiol/issue?id=10.1371/issue.pcbi.v%s.i%s'
    papers_info = {}
    papers_original = {}

    volumes = ['15', '14', '13', '12', '11', '10', '09', '08', '07', '06', '05']
    issues = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    for v in volumes:
        for i in issues:
            if v == '15':
                if i != '01' and i != '02':
                    continue

            print "Processing PLOS Computational Biology volume %s and issue %s .... " % (v, i)
            url = base_url % (v, i)
            print url

            sys.stdout.flush()
            try:
                response = urllib2.urlopen(url)
                soup_info = BeautifulSoup(response.read(), "html.parser")

                for div in soup_info.find_all('div', {'class': 'item cf'}):
                    for a_tag in div.find_all('a'):
                        _url = a_tag.get('href')

                        if _url.startswith('https://'):
                            r = urllib2.urlopen(_url)
                            soup_content = BeautifulSoup(r.read(), "html.parser")

                            div = soup_content.find("div", {'class': 'article-type'})
                            article_type = div.find('p').contents[0].strip()

                            if article_type == 'Research Article':
                                title = soup_content.find('title').text

                                year = int(v) + 2004
                                k = _url.split('.')[-1]
                                contents = ""
                                for c in soup_content.find_all("div", {'class': 'abstract toc-section'}):
                                    for p in c.find_all('p'):
                                        contents += p.text.strip()
                                        contents += ' '
                                for c in soup_content.find_all("div", {'class': 'section toc-section'}):
                                    for p in c.find_all('p'):
                                        contents += p.text.strip()
                                        contents += ' '

                                # keep valid record
                                if len(contents) >= 3000 and 2009 <= year <= 2019:
                                    papers_original[k] = contents
                                    papers_info[k] = {'title': title, 'url': _url, 'year': year}
                                    print papers_info[k], len(contents)

            except Exception:
                print "OOps: " + url
                continue

    print "%d papers are collected from PLOS Computational Biology" % len(papers_info)
    with open(BIO_PLOS_COMPBIO_INFO_PATH, 'w') as fp:
        json.dump(papers_info, fp)

    with open(BIO_PLOS_COMPBIO_ORI_PATH, 'w') as fp:
        json.dump(papers_original, fp)