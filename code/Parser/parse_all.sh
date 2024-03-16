#! /bin/bash
sdir=$SOURCEDIR
tdir=$TODIR

./preprocess.sh $sdir $tdir
./gencpg.sh $tdir
./getpdg.sh $tdir
