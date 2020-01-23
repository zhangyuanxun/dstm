from bs4 import BeautifulSoup
import urllib2
import json
from contants import *
import sys


def collector():
    base_url = "http://www.jneurosci.org/content/"

    papers_info = {}
    papers_original = {}
    print "Start to collect paper from Journal of Neuroscience "
    for v in range(29, 40):
        print "Collect paper from Journal of Neuroscience in the year of %d " % (v + 1980)
        sys.stdout.flush()
        for i in range(1, 60):
            url = base_url + str(v) + '/' + str(i)
            try:
                response = urllib2.urlopen(url)
                soup = BeautifulSoup(response.read(), "html.parser")

                # Get all the hrefs of this issues
                for a in soup.find_all('a', href=True):
                    template = '/content/' + str(v) + '/' + str(i) + '/'

                    if template in a['href'] and str.isdigit(str(a['href'].split(template)[1])):
                        paper_url = base_url + a['href'].split('/content/')[1]

                        r = urllib2.urlopen(paper_url)
                        s = BeautifulSoup(r.read(), "html.parser")

                        m = s.find("meta", {"name": "article:section"})['content']
                        if m == "Articles" or m == "Research Articles":
                            title = s.find("div", {"class": "highwire-cite-title"}).text
                            k = a['href'].split('/content/')[1]

                            response = urllib2.urlopen(paper_url)
                            soup_content = BeautifulSoup(response.read(), "html.parser")

                            for tag in soup_content.find_all('em'):
                                tag.replaceWith('')

                            contents = ""
                            for tag in soup_content.findAll('p'):
                                contents += tag.text
                            contents = contents.strip()

                            year = v + 1980
                            if len(contents) > 3000 and 2009 <= year <= 2019:
                                papers_original[k] = contents
                                papers_info[k] = {'title': title, 'url': paper_url, 'year': year}
            except Exception:
                continue

    print "%d papers are collected from Journal of Neuroscience " % len(papers_info)

    with open(NEURO_JON_INFO_PATH, 'w') as fp:
        json.dump(papers_info, fp)

    with open(NEURO_JON_ORI_PATH, 'w') as fp:
        json.dump(papers_original, fp)