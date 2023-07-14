# About LESS

## How to run it
1. Edit the "./config/LESS.yaml"

2. run 
    ```
    python LESS.py
    ```
3. In default, the output dir is set in the same dir of dataset, for example, "data/sequence/00/LESS/group" and "data/sequence/00/LESS/labels".

4. In default, we start the work from last completed work. If you want to start again, please remove the output dir mentioned above.


##  params 
To change the param, rewrite the "/config/LESS.yaml" 
```
LESS:
  RANSAC:
    scans_per_subsequence: 5
    l_grid: 5
    iter_max: 100
    percent: 0.8
    dist: 3
  cluster:
    d: 0.1
```
The params are divided into "RANSAC" and "cluster". The introduction of params is given below.

**scans_per_subsequence** means t scans in one subsequence

**l_grid** means divide the ground into l_grid*l_grid blocks

**iter_max** means the RANSAC stops if iter == iter_max, which is one of stop condition

**precent** is another stop condition, when the ground points >= percent * all_points

**dist** is the highest bar that a point is treated as a inner point(ground point)

we treat a point as ground point, when its distance to the fitting flat <  dist 

the cluster_d is the d in the paper, which is in the formula:

``
                τ (u, v) = max(ru, rv) × d
``


### procedure of RANSAC

In our work, RANSAC is used to reduce the ground points.

In the article "LESS", authors divide the sequence into many subsequences consisting several scans. 

The parameter "scan_per_subsequence" represents the number of scans in a subsequence.

We aggregate this scans with pose in "pose.txt". Then we divide the points by (x,y). The size of each block is defined by "l_grid". Our points is divided into (l_grid*l_grid) blocks. Then in each block, we run the RANSAC.

During RANSAC, we fit a flat with random three points, which is the least points for flat fitting.

Then we need to get the inner points. Inner points is the points whose distance to the fitting flat is less than the bar value "dist". So far, we finish an iter.

Then We run the iter again. We have two stop condition.

1. iter >= iter_max.
2. inner_points >= percent * points_number

### procedure of cluster

This part follow the method given in the papar. 

The parameter "d" represent the bar value of cluster.
