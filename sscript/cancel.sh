#!/bin/bash
for jobid in {33827154..33827161}
do
    scancel $jobid
done