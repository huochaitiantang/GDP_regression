# GDP_regression
Chinese GDP regression by  Least squares approximation 

## Dependency
`pip install -r requirements.txt`

## Run

`python stat_GDP.py full/backward/forward/forward_backward 2.94/4.28/7.88/other_float`
* type = full/backward/forward/forward_backward (Args for regression type)
* thresh = 2.94/4.28/7.88/... (Args for F_in and F_out)

## Generate Linear Data for test
`python generate_test_data.py`

## Visual
* For Test Data
![avatar](/pics/test_linear_full_solve_visual.png)

* For GDP Data
![avatar](/pics/GDP_full_solve_visual.png)
