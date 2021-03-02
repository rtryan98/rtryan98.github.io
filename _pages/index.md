---
layout: defaults/page
permalink: index.html
narrow: true
title: Welcome!
---
Welcome to my Blog!

<hr />

### Recent Posts

{% for post in site.posts limit:3 %}
{% include components/post-card.html %}
{% endfor %}
