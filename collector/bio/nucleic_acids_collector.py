from bs4 import BeautifulSoup
import urllib2
import json
import sys
from constants import *


def collector():
    base_url = 'https://academic.oup.com/nar/issue/%s/%s'
    base_paper_url = 'https://academic.oup.com'
    papers_info = {}
    papers_original = {}

    volumes = ['40', '41', '42', '43', '44', '45', '46', '47']
    issues = ['W1', 'D1']

    for v in volumes:
        for i in issues:
            if v == '47':
                if i != 'D1':
                    continue

            print "Processing Nucleic Acids Research volume %s and issue %s .... " % (v, i)
            url = base_url % (v, i)
            print url

            sys.stdout.flush()
            try:
                response = urllib2.urlopen(url)
                soup_info = BeautifulSoup(response.read(), "html.parser")

                match_s = '/nar/article/' + v + '/' + i
                for a_tag in soup_info.find_all('a'):
                    _href = a_tag.get('href')
                    _class = a_tag.get('class')
                    if _href is not None and match_s in _href and _class is None:
                        _title = a_tag.text
                        _url = base_paper_url + _href
                        _year = 2000 + int(v) - 28
                        r = urllib2.urlopen(_url)

                        soup_content = BeautifulSoup(r.read(), "html.parser")
                        div = soup_content.find("div", {'class': 'article-metadata-tocSections'})
                        _type = div.find('a').text

                        if _type == "Web Server issue" or _type == "Database Issue":
                            contents = ""
                            k = v + i + _url.split('/')[-1]
                            for p_tag in soup_content.findAll('p'):
                                if p_tag.get('class') is not None:
                                    continue

                                contents += p_tag.text
                                contents += ' '

                            # keep valid record
                            if len(contents) > 2000 and 2009 <= _year <= 2019:
                                papers_original[k] = contents
                                papers_info[k] = {'title': _title, 'url': _url, 'year': _year}
                                print papers_info[k], len(contents)

            except Exception:
                print "OOps: " + url
                continue

    print "%d papers are collected from Nucleic Acids Research" % len(papers_info)
    with open(BIO_NUCLEIC_INFO_PATH, 'w') as fp:
        json.dump(papers_info, fp)

    with open(BIO_NUCLEIC_ORI_PATH, 'w') as fp:
        json.dump(papers_original, fp)