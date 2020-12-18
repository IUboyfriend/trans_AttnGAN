# from spacy.language import Language
# import numpy as np
#
# @Language.factory('tensor2attr')
# class Tensor2Attr:
#     def __init__(self, name, nlp):
#         pass
#
#     def __call__(self, doc):
#         # When an object is received, the class will instantly pass
#         # the object forward to the 'add_attributes' method. The
#         # reference to self informs Python that the method belongs
#         # to this class.
#         self.add_attributes(doc)
#         return doc
#
#     def add_attributes(self, doc):
#         # spaCy Doc objects have an attribute named 'user_hooks',
#         # which allows customising the default attributes of a
#         # Doc object, such as 'vector'. We use the 'user_hooks'
#         # attribute to replace the attribute 'vector' with the
#         # Transformer output, which is retrieved using the
#         # 'doc_tensor' method defined below.
#         doc.user_hooks['vector'] = self.doc_tensor
#
#         # We then perform the same for both Spans and Tokens that
#         # are contained within the Doc object.
#         doc.user_span_hooks['vector'] = self.span_tensor
#         doc.user_token_hooks['vector'] = self.token_tensor
#
#         # We also replace the 'similarity' method, because the
#         # default 'similarity' method looks at the default 'vector'
#         # attribute, which is empty! We must first replace the
#         # vectors using the 'user_hooks' attribute.
#         doc.user_hooks['similarity'] = self.get_similarity
#         doc.user_span_hooks['similarity'] = self.get_similarity
#         doc.user_token_hooks['similarity'] = self.get_similarity
#
#     # Define a method that takes a Doc object as input and returns
#     # Transformer output for the entire Doc.
#     def doc_tensor(self, doc):
#         # Return Transformer output for the entire Doc. As noted
#         # above, this is the last item under the attribute 'tensor'.
#         # Average the output along axis 0 to handle batched outputs.
#         return doc._.trf_data.tensors[-1].mean(axis=0)
#
#     # Define a method that takes a Span as input and returns the Transformer
#     # output.
#     def span_tensor(self, span):
#         # Get alignment information for Span. This is achieved by using
#         # the 'doc' attribute of Span that refers to the Doc that contains
#         # this Span. We then use the 'start' and 'end' attributes of a Span
#         # to retrieve the alignment information. Finally, we flatten the
#         # resulting array to use it for indexing.
#         tensor_ix = span.doc._.trf_data.align[span.start: span.end].data.flatten()
#
#         # Fetch Transformer output shape from the final dimension of the output.
#         # We do this here to maintain compatibility with different Transformers,
#         # which may output tensors of different shape.
#         out_dim = span.doc._.trf_data.tensors[0].shape[-1]
#
#         # Get Token tensors under tensors[0]. Reshape batched outputs so that
#         # each "row" in the matrix corresponds to a single token. This is needed
#         # for matching alignment information under 'tensor_ix' to the Transformer
#         # output.
#         tensor = span.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]
#
#         # Average vectors along axis 0 ("columns"). This yields a 768-dimensional
#         # vector for each spaCy Span.
#         return tensor.mean(axis=0)
#
#     # Define a function that takes a Token as input and returns the Transformer
#     # output.
#     def token_tensor(self, token):
#         # Get alignment information for Token; flatten array for indexing.
#         # Again, we use the 'doc' attribute of a Token to get the parent Doc,
#         # which contains the Transformer output.
#         tensor_ix = token.doc._.trf_data.align[token.i].data.flatten()
#
#         # Fetch Transformer output shape from the final dimension of the output.
#         # We do this here to maintain compatibility with different Transformers,
#         # which may output tensors of different shape.
#         out_dim = token.doc._.trf_data.tensors[0].shape[-1]
#
#         # Get Token tensors under tensors[0]. Reshape batched outputs so that
#         # each "row" in the matrix corresponds to a single token. This is needed
#         # for matching alignment information under 'tensor_ix' to the Transformer
#         # output.
#         tensor = token.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]
#
#         # Average vectors along axis 0 (columns). This yields a 768-dimensional
#         # vector for each spaCy Token.
#         return tensor.mean(axis=0)
#
#     # Define a function for calculating cosine similarity between vectors
#     def get_similarity(self, doc1, doc2):
#         # Calculate and return cosine similarity
#         return np.dot(doc1.vector, doc2.vector) / (doc1.vector_norm * doc2.vector_norm)
#
# import spacy
# import torch
# if __name__ == "__main__":
#     # Load the spaCy model with the 'tensor2attr' component
#     text_encoder = spacy.load('en_core_web_trf')
#
#     # List of texts to process
#     sentence_list = ["This is the first text 123.", "And this is the second text.", "Finally, the third text."]
#     embs = list(text_encoder.pipe(sentence_list)) # create a list of doc
#
#
#     # for doc in embs:
#     #     for i in doc._.trf_data.last_hidden_layer_state:
#     #
#
#     print(embs[2]._.trf_data.last_hidden_layer_state[0].data.shape)
#     print(type(embs[2]._.trf_data.last_hidden_layer_state))
#     print(type(embs[2]._.trf_data.last_hidden_layer_state[0].data))
#     #
#     # print(sentence_vector)
#
#     # print(type(embs[2])) #<class 'spacy.tokens.doc.Doc'>
#     # print(type(embs[2]._.trf_data)) #<class 'spacy_curated_transformers.models.output.DocTransformerOutput'>
#     # print("Number of sentences:",  len(embs))
#
#     # extract the first sentence's sentence embedding
#     # print(len(embs[0]._.trf_data.tensors[0]))
#     #
#     # print(embs[0].vector)
#     # print(embs[0]._.trf_data.last_hidden_layer_state[0])
#     # print(embs[0]._.trf_data.tensors[1])
#
#
#
import torch
from transformers import BertTokenizer, BertModel
if __name__ == "__main__":

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_encoder = BertModel.from_pretrained("bert-base-uncased")
    # List of texts to process
    sentence_list = ["This.", "And this is the second text.", "Finally, the third text 1 1 1 1."]

    tokenized_inputs = tokenizer(
        sentence_list,
        padding=True,  # Pad to the longest sequence in the batch
        truncation=True,  # Truncate to max length of the model
        return_tensors='pt'  # Return PyTorch tensors
    )
    # Forward pass through the BERT model
    with torch.no_grad():
        outputs = text_encoder(**tokenized_inputs)

    last_hidden_states = outputs.last_hidden_state # you should use only 1: length word features, excluding CLS


    # sentence feature vectors
    sentence_feature = last_hidden_states[:, 0, :].cuda()  # Shape: (batch_size, hidden_size)

    word_features = last_hidden_states[:, 1:, :].cuda()  # Shape: (batch_size, sequence_length, hidden_size = 768)
    word_features = word_features.permute(0, 2, 1)
    print(word_features.shape)
    print(type(word_features))
