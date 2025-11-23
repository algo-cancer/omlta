# O-MLTED

This repository contains an implementation of omlta, the optimal multi-label tree alignment problem, and omltd, the optimal multi-label tree edit distance. The provided script computes both the minimum cost edit sequence for the alignment as well as the distance between two input trees.

## Input

omlta.py requires two text files each specifying one of the two input trees. The format of the text files is three columns separated by whitespace: a node id, the node's parent id, and a list of the labels of the node.

The root node id should be ROOT with parent None.

The labels should be comma-separated, no white space.  See example-tree1.txt and example-tree2.txt for an example.

Labels do not need to be shared between the two trees.  Any labels unique to only one of the input tree files are removed by the OMLTED.py algorithm in a pre-processing step.

## Usage

This script is made in Python 3. To run the omlta/omltd implementation:

```python omlta.py tree-file1 tree-file2```

Sample execution on the included example tree files:
```
C:\OMLTED python omlta.py example-tree1.txt example-tree2.txt
Checking k =   16
Finished k =  16
Minimum cost edit sequence:
[('nd', '6', ['mut13', 'mut14', 'mut15']), ('ne', '1', ['mut2']), ('ne', '1', ['mut2']), ('nd', '1', ['mut3']), ('nd', '1', ['mut12']), ('nd', '5', ['mut12']), ('nd', '5', ['mut3']), ('nd', '6', ['mut13', 'mut14', 'mut15'])]
Non-common labels deleted in pre-processing:  0
omltd:  10
Normalized omltd:  0.29411764705882354
```

## Output
The output includes four things. We discuss them in the order they are output (see example above).

1. A minimum cost edit sequence on the two input forests. The sequence is provided as a list of tuples.  The first element of the tuple corresponds to the type of edit.  'nd' - node deletion, 'ne' - node expansion, 'ld' - label deletion.  The second element of the tuple is the key of the node used to identify which node in the tree is being edited.  The third element of the tuple is the set of labels either being expanded or deleted.

2. The number of labels of the input forests that are unique to only one of the forests (not shared) and are deleted during a pre-processing step.

3. The omltd value of the two input forests, i.e., the number of labels deleted in the minimum cost edit sequence.


4. The normalized omltd distance, i.e., the omlta distance divided by the total number of labels in the two input forests after pre-processing. 
