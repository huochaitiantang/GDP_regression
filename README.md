# GDP_regression
Chinese GDP regression by  Least squares approximation 

## Dependency
`pip install -r requirements.txt`

## Run
* Type: args for regression type, default = full
```
full
backward
forward
forward_backward
```
* Thresh: args for F_in and F_out, default = 2.94
```
2.94
4.28
7.88
...
```
* Format
```
python stat_GDP.py [Type] [Thresh]
```
* Example
```
python stat_GDP.py
python stat_GDP.py full
python stat_GDP.py forward
python stat_GDP.py backward 4.28
python stat_GDP.py forward_backward 2.00
```

## Generate Linear Data for test
`python tools/generate_test_data.py`

## Visual
* Full Model For Test Data
![avatar](/pics/test_linear_full_solve_visual.png)

* Full Model For GDP Data
![avatar](/pics/GDP_full_solve_visual.png)
