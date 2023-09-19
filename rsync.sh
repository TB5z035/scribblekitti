sequences=(07 08 09 10)

for((i=0;i<11;i++));
do
    rsync -av /data22/tb5zhh/datasets/SemanticKITTI/sequences/${sequences[i]}/velodyne yujc@112.90.84.45:/data/discover-08/yujc/SEMANTICKITTI/sequences/${sequences[i]}/velodyne 
    rsync -av /data22/tb5zhh/datasets/SemanticKITTI/sequences/${sequences[i]}/labels yujc@112.90.84.45:/data/discover-08/yujc/SEMANTICKITTI/sequences/${sequences[i]}/labels 
    rsync -av /data22/tb5zhh/datasets/SemanticKITTI/sequences/${sequences[i]}/scribbles yujc@112.90.84.45:/data/discover-08/yujc/SEMANTICKITTI/sequences/${sequences[i]}/scribbles 
    rsync -av /data22/tb5zhh/datasets/SemanticKITTI/sequences/${sequences[i]}/LESS_230916_only_superpoint_propogated yujc@112.90.84.45:/data/discover-08/yujc/SEMANTICKITTI/sequences/${sequences[i]}/LESS_230916_only_superpoint_propogated 
    # cd /data/hdd01/pengfeili/first/semanticKITTI/dataset/sequences_1
    # mkdir ${sequences[i]}
    # cd /data/hdd01/pengfeili/first/semanticKITTI/dataset/sequences
    # mv ${sequences[i]}/velodyne /data/hdd01/pengfeili/first/semanticKITTI/dataset/sequences/${sequences[i]}/velodyne
done