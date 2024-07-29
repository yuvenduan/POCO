#!/bin/bash
for jobid in {41457973..41458044}
do
    scancel $jobid
done