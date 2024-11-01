# brain-match
Match two Connectome brain graphs.

This contains several helper programs as well:

**transform-connectome-graph**: Convert the VNC Connectome male and female graphs into generic graphs \
**transform-solution**: Convert a solution matching into a VNC Connectome male and female solution file \
**convert-VNC-matching**: Convert the VNC Connectome matching solution into a generic one \
**score-matching**: Find the score of a matching between two graphs \
**identity-match**: Given one graph, create a matching (i,i) for each vertex \
**spectral-match**: Given two graphs, use a spectral method to find a matching \
**greedy-match**: Given two graphs, use a greedy similarity method to find a matching \
**refine-match-hillclimb**: Given two graphs and matching, use hill-climbing to improve the match


**Sample workflow:** \
```
% gunzip female_connectome_graph.csv.gz \
% gunzip male_connectome_graph.csv.gz \
% transform-connectome_graph -f female_connectome_graph.csv \
% transform-connectome_graph -m male_connectome_graph.csv \
% greedy-match gm.csv gf.csv matching.csv \
% score-matching  gm.csv gf.csv matching.csv \
```

David A. Bader
dbader13@gmail.com

