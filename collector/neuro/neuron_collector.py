from bs4 import BeautifulSoup
import urllib2
import json
from constants import *
import sys


def collector():
    base_url = "https://www.cell.com"

    papers_info = {}
    papers_original = {}

    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor())
    front_page = '/neuron/archive'
    response = opener.open(base_url + front_page)
    soup_archives = BeautifulSoup(response.read(), "html.parser")

    archive_urls = []

    for a in soup_archives.find_all('a', href=True):
        url = a['href']

        if '/neuron/issue?pii=' in url and url not in archive_urls:
            archive_urls.append(url)
    print "Start to collect paper from NEURON "
    sys.stdout.flush()

    for url in archive_urls:
        volume_url = base_url + url

        print "Start to collect paper from volume  " + str(volume_url)
        sys.stdout.flush()
        response = opener.open(volume_url)
        soup_volume = BeautifulSoup(response.read(), "html.parser")

        for a in soup_volume.find_all('a', href=True):
            paper_url = a['href']
            if '/neuron/fulltext/' in paper_url:

                response = opener.open(base_url + paper_url)
                soup_content = BeautifulSoup(response.read(), "html.parser")

                label = soup_content.find("span", {'class': 'article-header__journal'}).contents[0]

                if label == 'Article':

                    title = soup_content.find("meta", {"name": "citation_title"})['content']
                    year = int(soup_content.find("meta", {"name": "citation_date"})['content'].split('/')[0])
                    k = paper_url.split('/')[-1]

                    if k in papers_info:
                        continue

                    contents = ""

                    # remove References
                    for div in soup_content.find_all("div", {'class': 'reference-citations__links'}):
                        div.decompose()

                    for div in soup_content.find_all("div", {'class': 'dropBlock reference-citations'}):
                        div.decompose()

                    for div in soup_content.find_all("div", {'class': 'figure__caption__body'}):
                        div.decompose()

                    for div in soup_content.find_all("div", {'class': 'section-paragraph supplemental-information'}):
                        div.decompose()

                    for div in soup_content.find_all('div', {'class': 'section-paragraph'}):
                        contents += div.text + " "

                    contents = contents.strip()
                    if len(contents) > 3000 and 2009 <= year <= 2019:
                        papers_original[k] = contents
                        papers_info[k] = {'title': title, 'url': paper_url, 'year': year}

    print "%d papers are collected from Neuron Journal " % len(papers_info)

    with open(NEURO_NEURON_INFO_PATH, 'w') as fp:
        json.dump(papers_info, fp)

    with open(NEURO_NEURON_ORI_PATH, 'w') as fp:
        json.dump(papers_original, fp)