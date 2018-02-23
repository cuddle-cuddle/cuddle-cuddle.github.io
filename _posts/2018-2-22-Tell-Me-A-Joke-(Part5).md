---
layout: post
title: "NLP: Short Sentence comparison: No NLP (Part 5: How are Trump Jokes like Cancer?)"
feature-img: "assets/img/posts/2018-02-22/trump_joke_together2.png"
thumbnail: "assets/img/posts/2018-02-22/trump_joke_together2.png"
---

# Conclusion: Trump Jokes are very much like cancer: they grow, then they stablize in a predictable manner. Like Cancer.
**Downloads**: [Jupyter Notebook: ]({{ site.baseurl }}/jupyter/trump_cancer_blog.ipynb)
[Data/CSV: ]({{ site.baseurl }}/jupyter/joke_score_name_clean.csv)
[Cancer Model:](https://en.wikipedia.org/wiki/Logistic_function#In_medicine:_modeling_of_growth_of_tumors)


In the last post, we looked at which politician is made fun of when.
In this one, we'll look at one particular person and one particular event and see how it effect the results.

... long story short, it turns out the a popular model, generalized logistic function is a decent model for the trump joke growth.

### Zoom out: Trump jokes have really picked up in popularity since election.(X:date, Y: tally of upvotes of jokes made against Trump, since 2015)
![Trump Jokes]({{ site.baseurl }}/assets/img/posts/2018-02-22/trump_jokes.png)

### Trump jokes on the week of Nov. 8th, 2016
![Election Week]({{ site.baseurl }}/assets/img/posts/2018-02-22/election.png)

## Mathematical Model:
### Trump jokes on the week of (blue: data, red: model)
![Election Week +6 months]({{ site.baseurl }}/assets/img/posts/2018-02-22/travel-ban2.png)

#### Bonus: what happened at the circled time? Why does the model not fit any more? SCROLL ALL THE WAY DOWN FOR THE ANSER!

The road towards the end results wasn't always straight forward, some things that I've tried but didn't work:
<ul>
<li>Ajit Pai: Attention span of people on that guy is too damn short</li>
<li>Number of jokes instead of scores: too hard to model</li>
<li>Rate of jokes per time unit: ... too boring </li>
</ul>
any how, enjoy the result: what works.

{% include_relative jupyter/trump_cancer_blog/trump_cancer_blog.md %}
