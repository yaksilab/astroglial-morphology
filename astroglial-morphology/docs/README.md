# Data structure for sub segmentation

The output data is stored two matrices, one for trace and one for coordinates of masks.


## Trace data structure
	
```math
\begin{bmatrix}
s2p & S_1 & f_1 & f_2 & ... & f_F \\ 
s2p & S_2 & f_1 & f_2 & ... & f_F \\
... \\
... \\
s2p & S_3 & f_1 & f_2 & ... & f_F \\
s2p & S_4 & f_1 & f_2 & ... & f_F \\
... \\
... \\

s2p & S_5 & f_1 & f_2 & ... & f_F \\
s2p & S_6 & f_1 & f_2 & ... & f_F \\
... \\
... \\
\end{bmatrix}
```

Where:
- S: Segment Label
- $f_1, f_2, ... f_F$: trace intensity across time points or frames


## Mask data structure

```math
\begin{bmatrix}
s2p & C_1 & S_1 & S_{\#x} & x_{o1} & y_{o1} & x_{a1} & y_{a1} & class \\
s2p & C_1 & S_1 & S_{\#x} & x_{o2} & y_{o2} & x_{a2} & y_{a2} & class \\
... \\
... \\
s2p & C_1 & S_2 & S_{\#x} & x_{o1} & y_{o1} & x_{a1} & y_{a1} & class \\
s2p & C_1 & S_2 & S_{\#x} & x_{o2} & y_{o2} & x_{a2} & y_{a2} & class \\
... \\
... \\
s2p & C_1 & S_3 & S_{\#x} & x_{o1} & y_{o1} & x_{a1} & y_{a1} & class \\
s2p & C_1 & S_3 & S_{\#x} & x_{o2} & y_{o2} & x_{a2} & y_{a2} & class \\
... \\
... \\
s2p & C_2 & S_4 & S_{\#x} & x_{o1} & y_{o1} & x_{a1} & y_{a1} & class \\
s2p & C_2 & S_4 & S_{\#x} & x_{o2} & y_{o2} & x_{a2} & y_{a2} & class \\
... \\
... \\
s2p & C_2 & S_5 & S_{\#x} & x_{o1} & y_{o1} & x_{a1} & y_{a1} & class \\
s2p & C_2 & S_5 & S_{\#x} & x_{o2} & y_{o2} & x_{a2} & y_{a2} & class \\
... \\
... \\
s2p & C_N & S_M & S_{\#x} & x_{o1} & y_{o1} & x_{a1} & y_{a1} & class \\
s2p & C_N & S_M & S_{\#x} & x_{o2} & y_{o2} & x_{a2} & y_{a2} & class \\
... \\

\end{bmatrix}
```

Where:
- $s2p$: suite2p roi index
- $C$: Cell Label in cellpose 
- $S$: Segment Label in cellpose 
- $S_{\#x}$: Subsegment number
- $(x_o, y_o)$: Orginal pixel coordinates
- $(x_a, y_a)$: Aligned pixel coordinates
- $class$: class label for the subsegment 
    - 1: complete cell upper part
    - 2: complete cell lower part