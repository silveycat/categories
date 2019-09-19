This repository contains three files:

1) `data.py` - all experiment data in Python format
2) `analysis_scripts.py` - all scripts for calculating the measures reported in the paper and writing to text files
3) `R_code.r` - R code that reads in these text files and outputs the confidence intervals, graphs, and statistical 
analyses reported in the paper

The repository also includes, in the output folder, pre-generated text files that can be directly analysed using the 
R code.

## Running the scripts
Please note that since part of the analysis involves a genetic algorithm, regenerating the text files from the Python 
script takes a long time - around 15 hours in total! To run the Python analysis script, you need to have Python 3 
installed. The script also uses [tqdm](https://pypi.org/project/tqdm/) to show progress bars for the time-intensive 
parts of the analysis (calculating convexity and convergence). To install tqdm and run the analysis, 
in the root directory run the following:
```
pip install tqdm
python analysis_scripts.py
```

Once the output files have been produced, the R script can be run in R or RStudio to produce the graphs and analyses.

The repository also includes the image stimuli used in the experiments (in the stimuli folder).

## Citation
> Silvey, C., Kirby, S., & Smith, K. (2019). Communication increases category structure and alignment only when combined with
 cultural transmission. *Journal of Memory and Language*, *109*, 104051. doi:10.1016/j.jml.2019.104051