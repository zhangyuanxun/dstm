from bs4 import BeautifulSoup
import urllib2
import json
from contants import *
import sys


def collect():
    base_url = "https://link.springer.com/journal/10827/"
    volume = range(26, 47)
    issues = range(1, 4)
    papers_info = {}
    papers_original = {}
    print "Start to collect paper from Journal of Computational Neuroscience "
    for v in volume:
        for i in issues:
            sys.stdout.flush()
            print "Processing Journal of Computational Neuroscience volume %d .... " % v
            if v == 46 and (i == 2 or i == 3):
                continue
            url = base_url + str(v) + '/' + str(i) + '/'

            try:
                response = urllib2.urlopen(url)
                soup = BeautifulSoup(response.read(), "html.parser")

                for link in soup.find_all('h3', {'class': 'title'}):
                    _url = link.find('a').get('href')
                    _title = link.find('a').contents[0].strip()
                    k = _url.split('/')[3]
                    _url = 'https://link.springer.com/' + _url

                    if 'Erratum' in _title:
                        continue

                    # # collect content from paper url
                    response = urllib2.urlopen(_url)
                    soup_content = BeautifulSoup(response.read(), "html.parser")

                    year = int(soup_content.find("meta", {"name": "citation_cover_date"})['content'].split('/')[0])

                    # remove equation
                    for span in soup_content.find_all("span", {'class': 'InlineEquation'}):
                        span.decompose()

                    # remove figure
                    for span in soup_content.find_all("span", {'class': 'InternalRef'}):
                        span.decompose()

                    # remove citation
                    for span in soup_content.find_all("span", {'class': 'CitationRef'}):
                        span.decompose()

                    contents = ""
                    for a in soup_content.find_all('p', {'class': 'Para'}):
                        contents += a.text

                    contents = contents.strip()

                    if len(contents) > 3000 and 2009 <= year <= 2019:
                        papers_original[k] = contents
                        papers_info[k] = {'title': _title, 'url': _url, 'year': year}

            except Exception:
                print "OOps: " + url
                continue

    print "%d papers are collected from Journal of Computational Neuroscience " % len(papers_info)

    with open(NEURO_JOCN_INFO_PATH, 'w') as fp:
        json.dump(papers_info, fp)

    with open(NEURO_JOCN_ORI_PATH, 'w') as fp:
        json.dump(papers_original, fp)