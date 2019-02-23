"""
Corpus reader

Reads the Corpus (from structured XML files) into memory.
"""

import xml.sax
import lxml.sax, lxml.etree
import re

class XmlDataExtractor(xml.sax.ContentHandler):
    """
    Generic XML data extractor.
    """

    def __init__(self):
        super().__init__()
        self.data = dict()
        self.lxmlhandler = None

    def get_data(self):
        return self.data


class NewsExtractor(XmlDataExtractor):
    """
    Parsing xml units with xml.sax + lxml.sax.ElementTreeContentHandler.
    Parser used in ../scripts/semeval-pan-2019-tf-extractor.py
    """

    def __init__(self):
        super().__init__()
        self.lxmlhandler = None

    def startElement(self, name, attrs):
        if name == 'articles':
            return
        if name == 'article':
            self.lxmlhandler = lxml.sax.ElementTreeContentHandler()
        self.lxmlhandler.startElement(name, attrs)

    def characters(self, data):
        if self.lxmlhandler is not None:
            self.lxmlhandler.characters(data)

    def endElement(self, name):
        if self.lxmlhandler is not None:
            self.lxmlhandler.endElement(name)
            if name =='article':
                # complete article parsed
                article = NewsArticle(self.lxmlhandler.etree.getroot())
                self.data[article.get_id()] = article
                self.lxmlhandler = None


class NewsExtractorFeaturizerFromStream(XmlDataExtractor):
    """
    Extracts news data from the XML stream and immediately featurizes
     each article, not needing to keep the whole dataset in memory.
    """

    def __init__(self, featurizer):
        super().__init__()
        self.lxmlhandler = None
        self.featurizer = featurizer
        self.counter = 0

    def startElement(self, name, attrs):
        if name == 'articles':
            return
        if name == 'article':
            self.lxmlhandler = lxml.sax.ElementTreeContentHandler()
        self.lxmlhandler.startElement(name, attrs)

    def characters(self, data):
        if self.lxmlhandler is not None:
            self.lxmlhandler.characters(data)

    def endElement(self, name):
        if self.lxmlhandler is not None:
            self.lxmlhandler.endElement(name)
            if name =='article':
                # complete article parsed
                self.counter += 1
                if self.counter % 500 == 0:
                    print('Progress: {:5}'.format(self.counter))

                article = NewsArticle(self.lxmlhandler.etree.getroot())
                self.data[article.get_id()] = self.featurizer(article)
                self.lxmlhandler = None


class GroundTruthExtractor(XmlDataExtractor):
    """
    SAX parser for gound truth XML.
    """

    def __init__(self):
        super().__init__()

    def startElement(self, name, attrs):
        if name == 'article':
            hyperpartisan = attrs.getValue('hyperpartisan')
            bias = attrs.getValue('bias') if 'bias' in attrs else None
            self.data[attrs.getValue('id')] = (hyperpartisan, bias)


class NewsArticle:
    """
    Class representing a single News article from the Hyperpartisan News Corpora.
    """

    def __init__(self, rootNode):
        self.root = rootNode
        self.text = lxml.etree.tostring(rootNode, method="text", encoding="unicode")
        
        self.hyperpartisan = None
        self.bias = None
    
    def get_id(self):
        return self.root.get('id')
    
    def get_title(self):
        return self.root.get('title')
    
    def get_text(self):
        return self.text
    
    def get_text_cleaned(self):
        return re.sub('[^A-Za-z ]', '', self.text)
    
    def set_ground_truth(self, hyperpartisan, bias):
        self.hyperpartisan = hyperpartisan
        self.bias = bias

    def get_hyperpartisan(self):
        assert self.hyperpartisan is not None
        return self.hyperpartisan

    def get_bias(self):
        ## May be None if dataset was labeled by article (not by publisher)
        # assert self.bias is not None
        return self.bias
