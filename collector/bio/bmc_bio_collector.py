from bs4 import BeautifulSoup
import urllib2
import json
import sys
from contants import *


def collector():
    base_url = 'https://bmcbioinformatics.biomedcentral.com/articles?searchType=journalSearch&sort=PubDate&page=%d'
    base_paper_url = 'https://bmcbioinformatics.biomedcentral.com'
    papers_info = {}
    papers_original = {}

    for i in range(1, 131):
        print "Processing BMC Bioinformatics page %d .... " % i
        url = base_url % i
        sys.stdout.flush()
        try:
            response = urllib2.urlopen(url)
            soup_info = BeautifulSoup(response.read(), "html.parser")
            for link in soup_info.find_all('h3', {'class': 'c-teaser__title'}):
                _url = base_paper_url + link.find('a').get('href')

                r = urllib2.urlopen(_url)
                soup_paper = BeautifulSoup(r.read(), "html.parser")
                year = 0
                for m in soup_paper.find_all("meta"):
                    if m.get('name') == 'dc.date':
                        year = int(m.get('content').split('-')[0])
                        break

                for m in soup_paper.find_all("meta"):
                    if "citation_article_type" in str(m) and (m.get('content').lower() == 'Methodology Article'.lower()\
                                                              or m.get('content').lower() == 'Research Article'.lower()\
                                                              or m.get('content').lower() == 'Research'.lower()):
                        try:
                            k = _url.split('/')[5]
                            _title = link.find('a').contents[0].strip()
                        except Exception:
                            print "OOps: " + _url
                            continue
                        if k in papers_info:
                            print "repeated key : %s" % k

                        # collect content from paper url
                        response = urllib2.urlopen(_url)
                        soup_content = BeautifulSoup(response.read(), "html.parser")

                        for div in soup_content.find_all("div", {'class': 'EquationContent'}):
                            div.decompose()

                        for div in soup_content.find_all("div", {'class': 'Table'}):
                            div.decompose()

                        contents = ""
                        for a in soup_content.find_all('p', {'class': 'Para'}):
                            contents += a.text

                        for a in soup_content.find_all('div', {'class': 'Para'}):
                            contents += a.text

                        contents = contents.strip()

                        # keep valid record
                        if len(contents) > 1000 and 2009 <= year <= 2019:
                            papers_original[k] = contents
                            papers_info[k] = {'title': _title, 'url': _url, 'year': year}
                            print papers_info[k], len(contents)
                        break

        except Exception:
            print "OOps: " + url
            continue

    print "%d papers are collected from BMC Bioinformatics" % len(papers_info)
    with open(BIO_BMCINFOR_INFO_PATH, 'w') as fp:
        json.dump(papers_info, fp)

    with open(BIO_BMCINFOR_ORI_PATH, 'w') as fp:
        json.dump(papers_original, fp)

