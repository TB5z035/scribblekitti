#!/bin/bash


To_Delete=LESS_230804_add_z_sort

for  ((i=0;i<10;i++))
do
    cd /data22/tb5zhh/datasets/SemanticKITTI/sequences/0$i
    rm -rf $To_Delete
done 
cd /data22/tb5zhh/datasets/SemanticKITTI/sequences/10
rm -rf $To_Delete

