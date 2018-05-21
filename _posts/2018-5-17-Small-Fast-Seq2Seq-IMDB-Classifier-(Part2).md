---
layout: post
title: "Small Fast Seq2Seq IMDB Classifier (Part1a: Tokenizing Statistics)"
feature-img: "assets/img/posts/2018-05-16/tok_test.jpg"
thumbnail: "assets/img/posts/2018-05-16/tok_test.jpg"
---


**Downloads**: [Jupyter Notebook: ]({{ site.baseurl }}/jupyter/blog_spm_lowercase_vocab_test.ipynb)


### This is the 1.5th post of a 3 part post to explore google's [Sentence Piece](https://github.com/google/sentencepiece)'s(SP) tokenizing power.

This one is optional. I'm just looking at distribution of words and their frequencies.

At the end, I have also included a little example of how each tokenizer tokenizes the first line of Milton's paradise lost differently.

{% include_relative jupyter/imdb_class_seq2seq/blog_spm_lower_case_vocab_test.md %}
