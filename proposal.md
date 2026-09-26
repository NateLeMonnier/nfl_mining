# Project Proposal

**1. Who is in our group?**
Nathan LeMonnier (u1420813) and Oker Ottesen (u1432526)

**2. What data do we plan to use and where do we plan to get it from?**
We plan to use the nfl_data_py python library, which contains data from multiple NFL datasets and will have everything that we need. As it is a Python library, we will be able to import it.

**What structure do we want to mine from the data?**
On top of roster position, NFL players can be further categorized into position style. These styles are loosely constructed from playstyle and physical attributes, but are hard to cleanly assign for every player. We want to mine clusters in NFL data that align with these styles, and find new clusters that might not have an official term.

**Why is our problem interesting?**
Categorizing teams more deeply than only roster positions will be interesting as it will allow us to find what combination of players leads to the most success. We could build team archetypes with past success and compare them to active teams.

**What is new with our project?**
While Towards Data Science has written a blog post on clustering players deeper than roster position, it focused mainly on quarterbacks and did not use the clusters to predict team success or find combinations of clusters that work well together. We will extend the analysis by looking at each position and finding successful combinations.
