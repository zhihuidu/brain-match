# brain-match
Match two Connectome brain graphs.

This contains several helper programs as well:

**analyze-graph**: Analyze an input graph \
**transform-male-female-VNC-inputs**: Convert the VNC Connectome male and female graphs and matching file into generic graphs \
**transform-solution**: Convert a solution matching into a VNC Connectome male and female solution file \
**convert-VNC-matching**: Convert the VNC Connectome matching solution into a generic one \
**score-matching**: Find the score of a matching between two graphs \
**identity-match**: Given one graph, create a matching (i,i) for each vertex \
**spectral-match**: Given two graphs, use a spectral method to find a matching \
**greedy-match**: Given two graphs, use a greedy similarity method to find a matching \
**greedy-feature-match**: Given two graphs, use a greedy similarity method with extended structural features to find a matching \
**refine-match-hillclimb**: Given two graphs and matching, use hill-climbing to improve the match \
**refine-twohop**: Given two graphs and matching, use swaps in 2-hop neighborhoods to improve the match \
**graph-alignment**: Given two graphs, matching, and ordering, use swaps to improve the match. \
**graph-alignment-optimized**: Given two graphs, matching, and ordering, use swaps to improve the match \
**compute-betweenness-order**: Given a graph, create an ordering using approx. betweenness centrality


**Sample workflow:** 
```
% gunzip female_connectome_graph.csv.gz
% analyze-graph -v female_connectome_graph.csv
% gunzip male_connectome_graph.csv.gz
% analyze-graph -v male_connectome_graph.csv
% transform-male-female-VNC-inputs male_connectome_graph.csv female_connectome_graph.csv vnc_matching_submission_benchmark_5154247_generic.csv
% // greedy-match gm.csv gf.csv matching.csv
% // refine-twohop gm.csv gf.csv matching.csv new-matching.csv
% // transform-solution new-matching.csv
% graph-alignment-optimized gm.csv gf.csv gm_12818795_finetune.csv gf_4322180_finetune.csv matching.csv new-matching-mf.csv
% convert-VNC-matching new-matching-mf.csv
% score-matching  gm.csv gf.csv new-matching-mf_generic.csv
```

David A. Bader
dbader13@gmail.com

