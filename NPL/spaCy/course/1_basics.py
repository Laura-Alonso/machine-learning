# 1. NPL OBJECT

# Import spaCy. 
import spacy # type: ignore

 
# You can use the nlp object like a function to analyze text. Includes language-specific rules for tokenization etc.
# It also includes language-specific rules used for tokenizing the text into words and punctuation. spaCy supports a variety of languages
# Contains the processing pipeline. At the center of spaCy is the object containing the processing pipeline. 
# We usually call this variable "nlp".
# You can use the nlp object like a function to analyze text.
nlp = spacy.blank("en")   # Create a blank English nlp object.



# 2. DOC OBJECT

## When you process a text with the nlp object, spaCy creates a Doc object – short for "document". 
# The Doc lets you access information about the text in a structured way, and no information is lost.
## The Doc behaves like a normal Python sequence by the way and lets you iterate over its tokens, or get a token by its index. 
# But more on that later!

# Created by processing a string of text with the nlp object
doc = nlp("Connell and Marianne grow up in the same small town in the west of Ireland, but the similarities end there. In school, Connell is popular and well-liked, while Marianne is a loner. But when the two strike up a conversation – awkward but electrifying – something life-changing begins. Normal People is a story of mutual fascination, friendship and love. It takes us from that first conversation to the years beyond, in the company of two people who try to stay apart but find they can ’t.")

# Iterate over tokens in a Doc
for token in doc:
    print(token.text)



# 3. TOKEN OBJECT

## https://course.spacy.io/doc.png
## Token objects represent the tokens in a document – for example, a word or a punctuation character.
## To get a token at a specific position, you can index into the doc.
## Token objects also provide various attributes that let you access more information about the tokens. For example, 
# the .text attribute returns the verbatim token text.

token = doc[0]
print(token.text)



# 4. SPAN OBJECT

## https://course.spacy.io/doc_span.png
## A Span object is a slice of the document consisting of one or more tokens. It's only a view of the Doc and doesn't 
# contain any data itself.
## To create a span, you can use Python's slice notation. For example, 1:3 will create a slice starting from the token 
# at position 1, up to – but not including! – the token at position 3.

# A slice from the Doc is a Span object
span = doc[1:3]

# Get the span text via the .text attribute
print(span.text)



# 5. LEXICAL ATTRIBUTES

# They refer to the entry in the vocabulary and don't depend on the token's context.
doc = nlp("It costs $5 five.")

## Here you can see some of the available token attributes:

# i is the index of the token within the parent document.
print("Index:   ", [token.i for token in doc])

# text returns the token text.
print("Text:    ", [token.text for token in doc])

# is_alpha, is_punct and like_num return boolean values indicating whether the token consists of alphabetic characters, 
# whether it's punctuation or whether it resembles a number. For example, a token "10" – one, zero – or the word "ten" – T, E, N.
print("is_alpha:", [token.is_alpha for token in doc])
print("is_punct:", [token.is_punct for token in doc])
print("like_num:", [token.like_num for token in doc])

## Example
# In this example, you’ll use spaCy’s Doc and Token objects, and lexical attributes to find percentages in a text. 
# You’ll be looking for two subsequent tokens: a number and a percent sign.

    # Use the like_num token attribute to check whether a token in the doc resembles a number.
    # Get the token following the current token in the document. The index of the next token in the doc is token.i + 1.
    # Check whether the next token’s text attribute is a percent sign ”%“.

# Process the text
doc = nlp(
    "In 1990, more than 60% of people in East Asia were in extreme poverty. "
    "Now less than 4% are."
)

# Iterate over the tokens in the doc
for token in doc:
    # Check if the token resembles a number
    if token.like_num:
        # Get the next token in the document
        next_token = doc[token.i + 1]
        # Check if the next token's text equals "%"
        if next_token.text == "%":
            print("Percentage found:", token.text)




# 6. TRAINED PIPELINES

## Some of the most interesting things you can analyze are context-specific: for example, whether a word is a verb or 
# whether a span of text is a person name.

## Trained pipeline components have statistical models that enable spaCy to make predictions in context. This usually includes 
# part-of speech tags, syntactic dependencies and named entities.

## Pipelines are trained on large datasets of labeled example texts.

## They can be updated with more examples to fine-tune their predictions – for example, to perform better on your specific data.

## What are trained pipelines?
#Models that enable spaCy to predict linguistic attributes in context
    # Part-of-speech tags
    # Syntactic dependencies
    # Named entities
# Trained on labeled example texts
    # Can be updated with more examples to fine-tune predictions


# A. TRAINED PIPELINES

## spaCy provides a number of trained pipeline packages you can download using the spacy download command. For example, 
# the "en_core_web_sm" package is a small English pipeline that supports all core capabilities and is trained on web text.
# OUT IN THE TERMINAL: python -m spacy download en_core_web_sm

# The spacy.load method loads a pipeline package by name and returns an nlp object.
nlp = spacy.load("en_core_web_sm")

# The package includes:
    # Provides the binary weights that enable spaCy to make predictions.
    # The vocabulary
    # Meta information about the pipeline 
    # Configuration file used to train it. It tells spaCy which language class to use and how to configure the processing pipeline.


# B. PREDICTING PART-OF-SPEECH TAGS

## Let's take a look at the model's predictions. 
# In this example, we're using spaCy to predict part-of-speech tags, the word types in context.

# 1. First, we load the small English pipeline and receive an nlp object.
## nlp = spacy.load("en_core_web_sm")

# 2. Next, we're processing the text "She ate the pizza".
doc = nlp("She ate the pizza")

# 3. For each token in the doc, we can print the text and the .pos_ attribute, the predicted part-of-speech tag.
for token in doc:
    # Print the text and the predicted part-of-speech tag
    print(token.text, token.pos_)

# Note! In spaCy, attributes that return strings usually end with an underscore – attributes without the underscore return an integer ID value.


# C. PREDICTING SYNTACTIC DEPENDENCIES

## Predict how the words are related. For example, whether a word is the subject of the sentence or an object.

# The .dep_ attribute returns the predicted dependency label.
# The .head attribute returns the syntactic head token. You can also think of it as the parent token this word is attached to.

for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)

# https://course.spacy.io/dep_example.png