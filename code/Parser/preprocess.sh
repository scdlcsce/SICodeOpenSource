#! /bin/bash

source_dir=$SOURCEDIR
src_dir=$TODIR

total=$(find $source_dir -name "*.c" | wc -l)
echo $total
count=0

for file in $(find $source_dir -name "*.c" -print)
do
	echo "[$count/$total]: $file"
    java -jar preprocess.jar $file $src_dir
	count=`expr $count + 1`
done
