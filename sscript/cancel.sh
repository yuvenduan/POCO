#!/bin/bash
for jobid in {11770890..11770962}
do
    scancel $jobid
done