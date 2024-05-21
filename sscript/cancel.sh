#!/bin/bash
for jobid in {29184472..29184703}
do
    scancel $jobid
done