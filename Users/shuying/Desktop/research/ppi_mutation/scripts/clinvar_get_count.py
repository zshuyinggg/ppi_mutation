# %%
import xml.etree.cElementTree as etree
from pandas.errors import ParserError
# get an iterable
myfile = '/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/clinvar/ClinVarFullRelease_00-latest.xml'
out = '/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/clinvar/organize_clinar.txt'
# context = ET.iterparse(myfile, events=('start', 'end'))

def fast_iter_count(context):
    """
    http://lxml.de/parsing.html#modifying-the-tree
    Based on Liza Daly's fast_iter
    http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    See also http://effbot.org/zone/element-iterparse.htm
    """
    n=0
    for event, elem in context:
        if n==0: root=elem
        n+=1
        # func(elem, *args, **kwargs)
        # It's safe to call clear() here because no descendants will be
        # accessed
        elem.clear()
        # Also eliminate now-empty references from the root node to elem
        # for ancestor in elem.xpath('ancestor-or-self::*'):
        #     while ancestor.getprevious() is not None:
        #         del ancestor.getparent()[0]
        del event
        root.clear()
    del context
    print(n)



context = etree.iterparse(myfile, events=('end',))
fast_iter_count(context)