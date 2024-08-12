#!/bin/bash
for jobid in {42095333..42097457}
do
    scancel $jobid
done