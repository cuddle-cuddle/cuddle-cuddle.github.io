---
layout: post
title: "Training CNN to straighten photos, without labeled image dataset"
feature-img: "assets/img/posts/2018-04-02/rotations.png"
thumbnail: "assets/img/posts/2018-04-02/rotations.png"
---

# Conclusion: You can have lots of fun with plain image data sets, without any labeling at all!
**Downloads**:
[PASCAL dataset:](http://host.robots.ox.ac.uk/pascal/VOC/)


PASCAl is a fun annotated dataset. It contains image data, with object identification and location. 
For example, 

![Pascal Dataset]({{ site.baseurl }}/assets/img/posts/2018-04-02/pascal.png)

Then I thought, why don't I have some fun with it? 

I have taken the PASCAL dataset and thrown away all annotation. Next I wrote a few custom transformers, so that the pictures will be rotated, and the y values are rotations.

The backbone of the model is pretrained resnet34. Add custom head, find optimum LR, freeze and unfreeze head and fit a few times.
After 5 minutes of training, I got some somewhat satisfying results.

![Result]({{ site.baseurl }}/assets/img/posts/2018-04-02/rotations.png)

Shocking how little time I need to do something cool!

Two things I'd like to point out: 
<ul>
<li>Any data set will do, really. As long as the pictures are reasonably justified. </li>
<li>Writing of data loader for pytorch is the thing that took me the most time. The building of model, fine tuning, finding hyper parameters took no time at all.</li>
</ul>

Similar things you can do at home: Stretch detection! Video processing! Yey! Fun!