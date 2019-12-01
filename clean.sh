#!/bin/bash

# Remove latex temp files
rm -f build/*
rm -f **/*.aux **/*.log **/*.out **/*synctex.gz

# Remove python compiled files
rm -f **/*.pyc
