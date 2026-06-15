The three zip files "30n-120mut.zip", "60n-240mut.zip", "120n-480mut.zip" included in the simulated_data folder contain all simulated mutation trees used in the manuscript for comparing OMLTA against existing tree measures Ancestor-Descendant accuracy, Different Lineage accuracy, CASet, DISC, Bourque, MLTED as well as simulated mutation trees used in the manuscript for reporting runtimes for OMLTA with varying mutation tree sizes. Using the instances in the "30n-120muts.zip" file, we compute all tree scoring measures mentioned above to compare and report the results in the manuscript. Using the instances contained in all three zip file, we furthermore report the runtimes of our OMLTA implementation in the manuscript.  We now describe how we generated the trees for each file.

#### 30n-120mut.zip:
As described in the manuscript, we generated 5 distinct tree topologies for tumor progression, each with 30 nodes and 120 mutations assigned to the non-root nodes of each tree. These five trees represent ground truth trees for our simulated mutation tree instances.  For each, we alter the tree by choosing 5 distinct nodes and move the mutations on each node to a randomly chosen non-root node within the same tree at a set distance d away. We repeat this process for five different values of d and 10 times per value of d.  In total there are 250 simulated instances contained in the "30n-120mut.zip" file.

#### 60n-120mut.zip: 
We perform the same process as before but for larger trees.  We generate 5 distinct tree topologies for tumor progression, each with 60 nodesa and 240 mutations assigned to the non-root nodes of each tree. For each, we alter the tree by choosing 10 distinct nodes and move the mutations on each to a randomly chosen non-root node within the same tree at a set distance d away. We repeat this process for five different values of d and 10 times per value of d.  In total there are 250 simulated instances contained in  perform the same process for trees with 60 nodes and 240 mutations, in which we choose 10 nodes whose mutations are moved per simulated instance, contained in the "60n-240mut.zip" folder. 

#### 120n-480mut.zip:
Finally, we generate 5 distinct tree topologies for tumor progression, each with 120 nodes and 480 mutations assigned to the non-root nodes of each tree. For each, we alter the tree by choosing 20 distinct nodes and move the mutations on each to a randomly chosen non-root node within the same tree at a set distance d away. We repeat this process for five different values of d and 10 times per value of d.  In total there are 250 simulated instances contained in  perform the same process for trees with 120 nodes and 480 mutations, in which we choose 10 nodes whose mutations are moved per simulated instance, contained in the "120n-480mut.zip" folder. 


#### Tree File formatting:
Within each zip folder is two folders, "ground_tr" and "mutated" corresponding to the 5 ground truth trees set and the 250 altered trees set, respectively.  Files in each folder have the following naming convention "simNo_[sim_number]-s_[node_size]-m_[mutation_count]" where:
  
  - sim_number represents which of the 5 ground truth trees the file corresponds to
  - node_size is the number of nodes in the tree
  - mutation_count is the number of mutations in the tree

In addition, trees in the "mutated folder" have an additional tag at the end of their name of the form "sample_[sample_number]" in which sample_number represents the simulation repeat number (i.e. which iteration of choosing a set of mutations to move from the corresponding ground truth tree).
