---
layout: post
title: "NLP: Short Sentence comparison: No NLP (Part 4: How much do we make fun of Trump?)"
feature-img: "assets/img/posts/2018-02-16/2016election.png"
thumbnail: "assets/img/posts/2018-02-16/2016election.png"
---

In the last post, we did some comparison between jokes.

This is a friendly light excursion to simple animation, a nice way to visualize our result.
TFIDF is used to figure out what are the intersting topics of the joke corpus. The answer is pretty intuitive:
<ul>
<li>Men v.s. Women</li>
<li>Polititians & Celebrities</li>
<li>Rabbi, Priest, other religious Joes</li>
</ul>

In the short future, I might write another blog on this, but not now.

### statistics of jokes of the election week, Nov. 8th, 2016
![Reddit Jokes]({{ site.baseurl }}/assets/img/posts/2018-02-16/2016election_short.png)

[Jupyter Notebook: ]({{ site.baseurl }}/jupyter/jobfair2018_blog.ipynb)

Source can be found here:

{% include_relative html/jokes_count.html %}
