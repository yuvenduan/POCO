#!/bin/bash
for jobid in {44370643..44370692}
do
    scancel $jobid
done