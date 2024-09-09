#!/bin/bash
for jobid in {45085397..45085676}
do
    scancel $jobid
done