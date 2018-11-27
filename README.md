# GDP_regression
Chinese GDP regression by  Least squares approximation 

## Dependency
`pip install -r requirements.txt`

## Run

`python stat_GDP.py type thresh`
* type = full or backward or forward or forward_backward (Args for regression type)
* thresh = 2.94 or 4.28 or 7.88 or others (Args for F_in and F_out)

## Generate Linear Data for test
`python generate_test_data.py`

## Visual
* Full Model For Test Data
![avatar](/pics/test_linear_full_solve_visual.png)

* Full Model For GDP Data
![avatar](/pics/GDP_full_solve_visual.png)
