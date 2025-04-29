#!/bin/bash
for jobid in {12686373..12686444}
do
    scancel $jobid
done