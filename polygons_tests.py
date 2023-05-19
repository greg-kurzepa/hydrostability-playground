# This file contains tests I ran for polygons.py.
# To run again, copy to bottom of that file.

# sam = Triangle(((0,0),(1,0),(0,3)))
# frodo = Triangle(((1,0),(0,3),(1,3)))
# fellowship = Shape((sam, frodo))

# l1 = Line(np.array([[0,0]]).T, np.array([[1,1]]).T)
# l2 = Line(np.array([[0,2]]).T, np.array([[1,0]]).T)

#T3
#----
# h, sub_shape, line = square.float_height(np.deg2rad(0), 0.1)
# print(f"height: {h}, submerged area: {sub_shape.area}")
# print(f"submerged centroid: {sub_shape.centroid}")

# print("Printing triangles:")
# for triangle in sub_shape.triangles:
#     print(triangle.x, "\n")
#----

#T1
#----
# theta = np.deg2rad(-45)
# line = Line(square.centroid, np.array([[np.cos(theta),np.sin(theta)]]).T)
# print(line.a.T[0], line.d.T[0])
# print(line.perp_line.T[0])

# line2 = Line(line.a+0.1*line.perp_line, line.d)
# print(line2.a.T[0], line2.d.T[0])
# print(line2.perp_line.T[0])

# print(square.get_area_below(line))
# print(square.get_area_below(line2))
#----

#T2
#----
# line = Line(np.array([[0,0.1]]).T, np.array([[3,-1]]).T)
# print(square.get_area_below(line))
# line2 = Line(np.array([[0,0.1]]).T, np.array([[3,1]]).T)
# print(square.get_area_below(line2))
# x = np.linspace(-5,5,100)
# y = np.linspace(-5,5,100)
# is_below = np.zeros(shape=(100,100))
# for idxi, i in enumerate(x):
#     for idxj, j in enumerate(y):
#         if (is_below[idxi][idxj] = )
# p = np.array([[3,1]]).T
# p2 = np.array([[-3,1]]).T
# print(is_below_line(line2, p))
# print(is_below_line(line2, p2))
#----

# print(square.float_angle(0.9)[0])


## the below went in shape.float_angle()

# t = np.linspace(-np.pi/4, np.pi/4, 181)
# potentials = []
# potentials2 = []
# for theta in t:
#     h, sub_shape, line = self.float_height(theta, relative_density)
#     # potentials.append(relative_density * self.area * 9.81 * h - sub_area * (np.dot(sub_shape.centroid.T[0]-self.centroid.T[0], line.perp_line.T[0]) - h))
#     potentials.append(relative_density * self.area * 9.81 * h)
#     potentials2.append(9.81 * sub_shape.area * (np.dot(sub_shape.centroid.T[0]-self.centroid.T[0], line.perp_line.T[0]) + h))
    # if np.isclose(theta, 0):
    #     print(f"stuff: {sub_shape.centroid.T[0]}, {self.centroid.T[0]}, {sub_shape.centroid.T[0]-self.centroid.T[0]}, {line.perp_line.T[0]}")
    #     print(f"theta: {np.rad2deg(theta)}deg, d: {np.dot(sub_shape.centroid.T[0]-self.centroid.T[0], line.perp_line.T[0])}, h: {h}")
    #     print(f"potential block: {potentials[-1]}, potential water: {potentials2[-1]}")
    #     input()
# potentials = np.array(potentials)
# potentials2 = np.array(potentials2)
# potentials -= potentials.min()
# potentials2 -= potentials2.min()
# plt.plot(t, potentials, label="block")
# plt.plot(t, potentials2, label="water")
# plt.plot(t, potentials - potentials2, label="block - water")
# plt.legend()
# plt.show()