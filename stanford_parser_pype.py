import jpype
import util

class StanfordParser(object):
    def __init__(self):
        #~ jpype.startJVM('C:/Program Files (x86)/Java/jdk1.6.0_21/jre/bin/client/jvm.dll', '-Xmx800m')
        #~ jpype.startJVM(jpype.getDefaultJVMPath())
        jpype.startJVM('/usr/lib/jvm/java-6-openjdk/jre/lib/amd64/server/libjvm.so', '-Xmx2000m')

        nlpPackage = jpype.JPackage('edu').stanford.nlp
        self.lexicalizedParser = nlpPackage.parser.lexparser.LexicalizedParser('englishPCFG.ser.gz')
        self.grammaticalStructureFactory  = nlpPackage.trees.PennTreebankLanguagePack().grammaticalStructureFactory()

    def apply(self, sentence):
        tree = self.lexicalizedParser.apply(sentence)
        pos = tree.taggedYield()
        #~ dependencies = self.grammaticalStructureFactory.newGrammaticalStructure(tree).typedDependencies()
        #~ dependencies = self.grammaticalStructureFactory.newGrammaticalStructure(tree).typedDependenciesCollapsed(True)
        dependencies = self.grammaticalStructureFactory.newGrammaticalStructure(tree).typedDependenciesCCprocessed(True)

        return pos, tree, dependencies

    def __del__(self):
        jpype.shutdownJVM()

parser = None

def parse(sentence):
    global parser
    if not parser:
        parser = StanfordParser()
    return parser.apply(sentence)

def test_parser():
    text = """The first thing to see,
looking away over the water, was a kind of dull line, that was the
woods on t' other side; you couldn't make nothing else out; then
a pale place in the sky; then more paleness spreading around; then the
river softened up away off, and warn't black any more, but gray; you
could see little dark spots drifting along ever so far away
(trading scows, and such things; and long black streaks) rafts;
sometimes you could hear a sweep screaking; or jumbled up voices,
it was so still, and sounds come so far; and by and by you could
see a streak on the water which you know by the look of the streak
that there's a snag there in a swift current which breaks on it
and makes that streak look that way; and you see the mist curl
up off of the water, and the east reddens up, and the river, and
you make out a log cabin in the edge of the woods, away on the
bank on t'other side of the river, being a wood yard, likely, and
piled by them cheats so you can throw a dog through it anywheres;
then the nice breeze springs up, and comes fanning you from over
there, so cool and fresh and sweet to smell on account of the woods
and the flowers; but sometimes not that way, because they've left
dead fish laying around, gars and such, and they do get pretty
rank; and next you've got the full day, and everything smiling in
the sun, and the song birds just going it!"""
    lim = 125
    text1 = ' '.join(text.split(' ')[0:150])
    text2 = ' '.join(text.split(' ')[0:30])
    try:
        print parse(text1)
    except jpype._jexception.JavaException as e:
        print e
        print '! Sentence too long, skipping.'
    try:
        print parse(text2)
    except jpype._jexception.JavaException as e:
        print e
        print '! Sentence too long, skipping.'

def tmp():
    text = "Transport Canada Publication 312 serves as the authoritative document for airport specifications for land airports in Canada"
    text = "The pilot was certified and qualified"
    pos, tree, dependencies = parse(text)
    print dependencies

if __name__ == '__main__':
    #~ tmp()
    for output in parse("Immediately after the second touchdown, the pilot decided to perform a go-around."): print output
    #~ test_parser()
    #~ pos, tree, dependencies = parse('A review of the radar data indicated that the aircraft approached the Liverpool airport from the east, turned south across runway 25/07 and joined the circuit left-hand downwind for runway 25.')
    #~ print pos[10].word()
    #~ print pos[10].tag()
    #~ print pos
    #~ print dependencies
    #~ print dependencies[0].reln()
    #~ print dependencies[0].gov().value()
    #~ print dependencies[0].dep().index()
    #~ print
    #~ print tree
