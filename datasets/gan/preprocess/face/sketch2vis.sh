#!/bin/bash
rm ~/DataSet/GAN/SKETCH2VIS/val/imageA/*
rm ~/DataSet/GAN/SKETCH2VIS/val/imageB/*
rm ~/DataSet/GAN/SKETCH2VIS/train/imageA/*
rm ~/DataSet/GAN/SKETCH2VIS/train/imageB/*
ls -l /home/donny/DataSet/facegan/preprocess_sketch/aligned_db/144_aligned_images/train_A/ | grep jpg | awk  'BEGIN{srand();}{value=int(rand()*10000000); print value"\3"$NF }' | sort | awk -F"\3" '{print $2}' > sort.txt
cat sort.txt | grep jpg | awk 'FNR<=500{print "cp /home/donny/DataSet/facegan/preprocess_sketch/aligned_db/144_aligned_images/train_A/"$NF" ~/DataSet/GAN/SKETCH2VIS/val/imageA/"}'  > valA.sh
cat sort.txt | grep jpg | awk 'FNR<=500{print "cp /home/donny/DataSet/facegan/preprocess_sketch/aligned_db/144_aligned_images/train_B/"$NF" ~/DataSet/GAN/SKETCH2VIS/val/imageB/"}'  > valB.sh
cat sort.txt | grep jpg | awk 'FNR>500{print "cp /home/donny/DataSet/facegan/preprocess_sketch/aligned_db/144_aligned_images/train_A/"$NF" ~/DataSet/GAN/SKETCH2VIS/train/imageA/"}'  > trainA.sh
cat sort.txt | grep jpg | awk 'FNR>500{print "cp /home/donny/DataSet/facegan/preprocess_sketch/aligned_db/144_aligned_images/train_B/"$NF" ~/DataSet/GAN/SKETCH2VIS/train/imageB/"}'  > trainB.sh
rm sort.txt
sh valA.sh
sh valB.sh
sh trainA.sh
sh trainB.sh
rm valA.sh valB.sh trainA.sh trainB.sh
