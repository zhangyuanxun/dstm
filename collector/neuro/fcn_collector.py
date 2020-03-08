from bs4 import BeautifulSoup
import urllib2
import json
from constants import *
import sys


def fill_zeros(number):
    d = 5 - len(number)
    return '0' * d + number


def collector():
    papers_info = {}
    papers_original = {}
    base_url = "https://www.frontiersin.org/articles/10.3389/fncom."
    print "Start to collect paper from Frontier of Computational Neuroscience "
    for year in range(2010, 2020):
        print "Collect paper from FCN in the year of %d " % year
        sys.stdout.flush()
        for i in range(1, 200):
            url = base_url + str(year) + "." + fill_zeros(str(i))
            try:
                response = urllib2.urlopen(url)
                soup = BeautifulSoup(response.read(), "html.parser")
                _type = soup.find("h2").contents[0].strip()
                _title = soup.find("title").contents[0].strip().split('|')[1].strip()
                if _type == 'Original Research ARTICLE' or _type == 'Methods ARTICLE':
                    k = str(year) + "." + fill_zeros(str(i))

                    # get text
                    response = urllib2.urlopen(url)
                    soup_content = BeautifulSoup(response.read(), "html.parser")

                    # remove References
                    for div in soup_content.find_all("div", {'class': 'References'}):
                        div.decompose()

                    # remove equation
                    for div in soup_content.find_all("div", {'class': 'equationImageholder'}):
                        div.decompose()
                    [s.extract() for s in soup_content('math')]

                    contents = soup_content.find('div', attrs={"class": "JournalFullText"}).text
                    contents = contents.strip()

                    if len(contents) > 3000 and 2009 <= year <= 2019:
                        papers_original[k] = contents
                        papers_info[k] = {'title':_title, 'url': url, 'year': year}

            except Exception:
                continue

    print "%d papers are collected from Frontier of Computational Neuroscience " % len(papers_info)

    with open(NEURO_FCN_INFO_PATH, 'w') as fp:
        json.dump(papers_info, fp)

    with open(NEURO_FCN_ORI_PATH, 'w') as fp:
        json.dump(papers_original, fp)