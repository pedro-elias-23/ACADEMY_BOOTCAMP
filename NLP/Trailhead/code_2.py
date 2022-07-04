import stanza

nlp1 = stanza.Pipeline(lang='en', processors='tokenize')
doc1 = nlp1('This is a test sentence for stanza. This is another sentence.')
for i, sentence1 in enumerate(doc1.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence1.tokens], sep='\n')

print([sentence.text for sentence in doc1.sentences])


nlp2 = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)
doc2 = nlp2('This is a sentence.\n\nThis is a second. This is a third.')
for j, sentence2 in enumerate(doc2.sentences):
    print(f'====== Sentence {j+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence2.tokens], sep='\n')


nlp3 = stanza.Pipeline(lang='en', processors='tokenize', tokenize_pretokenized=True)
doc3 = nlp3('This is token.ization done my way!\nSentence split, too!')
for k, sentence3 in enumerate(doc3.sentences):
    print(f'====== Sentence {k+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence3.tokens], sep='\n')



nlp4 = stanza.Pipeline(lang='en', processors='tokenize', tokenize_pretokenized=True)
doc4 = nlp4([['This', 'is', 'token.ization', 'done', 'my', 'way!'], ['Sentence', 'split,', 'too!']])
for l, sentence4 in enumerate(doc4.sentences):
    print(f'====== Sentence {l+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence4.tokens], sep='\n')