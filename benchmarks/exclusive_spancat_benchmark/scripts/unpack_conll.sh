#!/bin/bash
mkdir temp
tar -xvf assets/conll.tgz -C temp
gzip -d temp/ner/data/esp.testa.gz
gzip -d temp/ner/data/esp.testb.gz
gzip -d temp/ner/data/esp.train.gz
gzip -d temp/ner/data/ned.testa.gz
gzip -d temp/ner/data/ned.testb.gz
gzip -d temp/ner/data/ned.train.gz
iconv -f iso-8859-1 temp/ner/data/esp.testa -o assets/es-conll-dev.iob
iconv -f iso-8859-1 temp/ner/data/esp.testb -o assets/es-conll-test.iob
iconv -f iso-8859-1 temp/ner/data/esp.train -o assets/es-conll-train.iob
iconv -f iso-8859-1 temp/ner/data/ned.testa -o assets/nl-conll-dev.iob
iconv -f iso-8859-1 temp/ner/data/ned.testb -o assets/nl-conll-test.iob
iconv -f iso-8859-1 temp/ner/data/ned.train -o assets/nl-conll-train.iob
rm -rf temp
rm assets/conll.tgz