---
layout: defaults/page
permalink: index.html
narrow: true
title: Welcome!
---

## What is it?

{% include components/intro.md %}
Welcome to my Blog!

<hr />

### Recent Posts

{% for post in site.posts limit:3 %}
{% include components/post-card.html %}
{% endfor %}


