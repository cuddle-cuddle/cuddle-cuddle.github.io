---
layout: post
title: "Small Fast Seq2Seq IMDB Classifier (Part1: Tokenizing)"
feature-img: "assets/img/posts/2018-05-16/8k_thick.jpg"
thumbnail: "assets/img/posts/2018-05-16/8k_thick.jpg"
---

## Conclusion: Sentence Piece is much better at tokenizing, retaining all information even when word is not in vocab.

**Downloads**: [Jupyter Notebook: ]({{ site.baseurl }}/jupyter/blog_spm_tokenizer_8k-lowercase.ipynb)


### This is the first post of a 3 part post to explore google's [Sentence Piece](https://github.com/google/sentencepiece)'s(SP) tokenizing power.

In a nut shell, this is the difference between SP's tokenizing effort and traditional approach:

_ | sentencepiece | do-it-you-self
unkown words (e.g. poopbutt) | UNKWON | tokenizes to poop+butt
saving vocabulary and model | DIY | does it for you
multi-threading  | DIY | does it for you  
util functions sich as String2Index | DIY | does it for you
tokenizing speed | depends on implementation | surprisingly fast!

Once again, google has delivered a lovely lovely useful tool.

{% include_relative jupyter/imdb_class_seq2seq/blog_spm_tokenizer_8k-lowercase.md %}
