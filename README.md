# GDP_regression
Chinese GDP regression by  Least squares approximation 

## Dependency
`pip install -r requirements.txt`

## Run

* type = full, backward, forward, forward_backward (Args for regression type, default = full)
* thresh = 2.94, 4.28, 7.88, ...(Args for F_in and F_out, default = 2.94)

`python stat_GDP.py [type] [thresh]`

* Example
`python stat_GDP.py`
`python stat_GDP.py full`
`python stat_GDP.py forward`
`python stat_GDP.py backward 4.28`
`python stat_GDP.py forward_backward 2.00`

## Generate Linear Data for test
`python generate_test_data.py`

## Visual
* Full Model For Test Data
![avatar](/pics/test_linear_full_solve_visual.png)

* Full Model For GDP Data
![avatar](/pics/GDP_full_solve_visual.png)
