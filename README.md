# SEMA
An Extended Semantic Evaluation Metric for AMR

# Requirements
Python (version 2.7 or later)

# Usage
`python sema.py -t parsed_file.txt -g reference_file.txt`

# Input files format
Test and Reference AMR files should be in PENMAN format.
```
(t / tolerate-01
    :ARG0 (w / we)
    :ARG1 (c / coutry
        :name (n / name :op1 "Japa")
        :wiki "Japa")
    :duration (a / amr-unknown))
```
