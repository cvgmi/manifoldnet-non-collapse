# Shows non-equivalence of 2 layer wFM to 1 layer wFM on SO(3)


import numpy as np
from geomstats.geometry import special_orthogonal


def generateWithinGeodesicBall(manifold, radius):
    d = distanceFunction(manifold)
    I = manifold.get_identity(point_type='matrix')

    while True:
        sample = manifold.random_uniform(point_type='matrix')[0]
        if (d(sample, I) < radius):
            return sample

def wFMFunction(manifold):
    def wFM(points, weights):
        points = np.stack(points)
        weights = np.array(weights)
        Xopt = manifold.exponential_barycenter(points, weights, point_type='matrix')
        return Xopt[0]
    
    return wFM

def distanceFunction(manifold):
    def d(X,Y):
        # transform X,Y to rotation vectors
        X_rv = manifold.rotation_vector_from_matrix(X)
        Y_rv = manifold.rotation_vector_from_matrix(Y)

        return manifold.bi_invariant_metric.dist(X_rv,Y_rv)[0][0]
    
    return d

# forward pass through two layer network with constant weights
def forwardTwoLayer(A,B,C, manifold):
    wFM = wFMFunction(manifold)
    h0 = wFM([A, B], [0.5, 0.5])
    h1 = wFM([B, C], [0.5, 0.5])

    return wFM([h0, h1], [0.5, 0.5])

def forwardOneLayer(A,B,C, w_1, w_2, manifold):
    wFM = wFMFunction(manifold)
    return wFM([A,B,C], [w_1, w_2, 1-(w_1+w_2)])
    

# finds weights w_1, w_2 s.t. X = wFM(A,B,C,w_1,w_2,1-w_2-w_1)
def optimizeWeights(X, A, B, C, w_1, w_2, manifold, threshhold=0.000000001, epsilon=0.01, verbose=False):
    # init subroutines for the manifold
    d = distanceFunction(manifold)
    wFM = wFMFunction(manifold)

    X_o = wFM([A,B,C], [w_1,w_2,1-(w_1+w_2)])
    loss_history = []

    while d(X, X_o) > threshhold:
        deltas = [[0,epsilon], [epsilon,0], [0, -epsilon], [-epsilon,0], [epsilon, epsilon], [-epsilon, -epsilon]]

        #update w_i
        delta_list = []
        for delta in deltas:
            X_o = wFM([A,B,C], [w_1+delta[0],w_2+delta[1], 1-(w_1+delta[0]+w_2+delta[1])])
            delta_list.append(d(X,X_o))
        min_delta = deltas[delta_list.index(min(delta_list))]

        # keep track of loss for lowering learning rate (epsilon)
        loss_history.append(min(delta_list))

        if len(loss_history) > 20 and loss_history[-3] == loss_history[-1]:
            epsilon = epsilon/2

        if(verbose):
            print("Distance to X: ", min(delta_list))
            print("Direction of travel: ", min_delta)

        w_1 += min_delta[0]
        w_2 += min_delta[1]

    return w_1, w_2, loss_history[-1]




if __name__ == "__main__":

    #we are working of SO(3)
    manifold = special_orthogonal.SpecialOrthogonal(3)
    #injectivity radius
    r = 0

    A = generateWithinGeodesicBall(manifold, r)
    B = generateWithinGeodesicBall(manifold, r)
    C = generateWithinGeodesicBall(manifold, r)

    X = forwardTwoLayer(A,B,C, manifold)

    #we will initialize with uniform weights
    w_1 = 0.3333
    w_2 = 0.3333


    print("Optimizing weights of one layer network...")
    w0, w1, d = optimizeWeights(X, A, B, C, w_1, w_2, manifold=manifold, verbose=False)
    weights = (w0, w1)

    print("Result: ", weights)

    print("Testing different network inputs")    

    distance_function = distanceFunction(manifold)
    distances = []
    for i in range(500):
        A = generateWithinGeodesicBall(manifold, r)
        B = generateWithinGeodesicBall(manifold, r)
        C = generateWithinGeodesicBall(manifold, r)

        out_two_layer = forwardTwoLayer(A,B,C, manifold)
        out_one_layer = forwardOneLayer(A,B,C, weights[0], weights[1], manifold)

        distances.append(distance_function(out_two_layer, out_one_layer))
    
    mean = sum(distances)/len(distances)
    print("Geodesic distance between Two Layer and One Layer networks at X: ", d)
    print("Mean Geodesic Distance between Two Layer and One Layer networks for random samples: ", mean)





