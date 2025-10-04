---
layout: defaults/page
permalink: index.html
narrow: true
title: Robert Ryan - Graphics Programmer
---

{% include components/intro.md %}

<hr />

### Recent Posts

{% for post in site.posts limit:3 %}
{% include components/post-card.html %}
{% endfor %}
