import numpy as np

# Step 3: Fit a linear regression model
slope, intercept = np.polyfit(np.array([4, 1, 2, 3, 4]), np.array([0, 2, 4, 6, 8]), 1)

# Step 5: Formulate the regression equation
regression_equation = f'y = {slope:.2f} * x + {intercept:.2f}'
print("Regression equation:", regression_equation)