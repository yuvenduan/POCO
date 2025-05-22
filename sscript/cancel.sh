#!/bin/bash
for jobid in {14613909..14614909}
do
    scancel $jobid
done