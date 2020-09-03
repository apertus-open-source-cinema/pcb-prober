#!/usr/bin/env python3
#
# Copyright (C) 2020 Herbert Poetzl

from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as plt


# original points

P = np.array([[-4, -2, 1], [4, -2, 1], [4, 2, 1], [-4, 2, 1]])

# transformation

T = [0.5, 0.6, 0.7]
R = t3d.euler.euler2mat(0.1, 0.2, 0.3, 'sxyz')
Z = [0.5, 0.4, 0.3]

# transformation matrix

A = t3d.affines.compose(T,R,Z)

# transformed points

Q = np.dot(P, A[0:3,0:3]) + A[0:3,3]



# calculate matrix from points

n = P.shape[0]
pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
unpad = lambda x: x[:,:-1]

X = pad(P)
Y = pad(Q)

B, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)

trans = lambda x: unpad(np.dot(pad(x), B))



# create some new points

p = np.array([(P[_]+P[(_+1)%4])/2 for _ in range(4)])

# transform those points

q = trans(p)




# visualize results

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(P[:,0], P[:,1], P[:,2], color='red')
ax.scatter(p[:,0], p[:,1], p[:,2], color='orange')

ax.scatter(Q[:,0], Q[:,1], Q[:,2], color='blue')
ax.scatter(q[:,0], q[:,1], q[:,2], color='teal')

D = Q-P
d = q-p

ax.quiver(P[:,0], P[:,1], P[:,2], D[:,0], D[:,1], D[:,2], color='pink', arrow_length_ratio=0.05)
ax.quiver(p[:,0], p[:,1], p[:,2], d[:,0], d[:,1], d[:,2], color='gold', arrow_length_ratio=0.05)

plt.show()

exit(0)

