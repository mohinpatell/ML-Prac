from numpy import *

def compute_error_for_line_given_points(b, m, points):
    # initialize it at 0
    totalError = 0
    # for every point
    for i in range(0, len(points)):
        # get x value
        x = points[i, 0]
        # get y value
        y = points[i, 1]

        # get difference, square it, add it to the total
        totalError += (y - (m * x + b)) ** 2
    # get the average
    return totalError / float(len(points))
        
def step_gradient(b_current, m_current, points, learningRate):
    # starting points for our gradients
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    # for every point in our data
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # direction with respect to b and m
        # computing partial derivatives of our error function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    # update our b and m values using our partial derivatives
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # starting b and m
    b = starting_b
    m = starting_m
    # gradient descent
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():

    # first collect the data
    points = genfromtxt("data.csv", delimiter=",")
    # x is hours studied and y is test score

    # define hyperparameters
    # how fast should our model converge
    # if it's too small, we'll get slow convergence
    # if it's too big, then our error function might not decrease
    learning_rate = 0.0001
    # y = mx + b
    initial_b = 0
    initial_m = 0
    # number of iterations to run the gradient descent
    num_iterations = 1000

    # now train the model
    print ('starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m,
                                                                              compute_error_for_line_given_points(
                                                                                  initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print ('ending point at b = {1}, m = {2}, error = {3}'.format(num_iterations, b, m, 
                                                                    compute_error_for_line_given_points(b, m, points))
)

if __name__ == "__main__":
    run()