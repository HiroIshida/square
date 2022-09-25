## Square
A toy square shape world to try path planning algorithms.

### Installation
```bash
pip3 install -e .
```

### Run example
```bash
python3 example.py
```
shows the following plot. The contour shows the signed distance function value and the black close line is the contour of f=0. Red lines shows the solution of the rrt planner. Blue lines shows the SQP based trajectory optimizer by using the rrt path as the initial solution.
![opt_result](https://user-images.githubusercontent.com/38597814/192170703-aae7bc23-785d-4889-8411-72d7d64f63b2.png)
