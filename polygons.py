# Written by Greg Kurzepa.
# PLEASE NOTE, this code is disgusting. It's badly orgnanised and, I don't doubt, practically undreadable.
# The task was complicated and I didn't want to faff with proper organisation. Extra unexpected challenges and edge cases kept
# adding nasty for loops and if statements, and probably made me pay for that decision.
#
# Other things that caused me problems that I probably shouldn't have done:
# Trying to store vectors as column vectors in numpy (i.e. meaning I had to add an extra dimension and transpose every numpy array I made...)
# Using python instead of C++.. oh god it's slow.. newton rhapson to the rescue I guess.
# Rotating the waterline instead of the object itself.. this also rotates the entire coordinate system, commence lots of unnecessary dot products.

import numpy as np
import matplotlib.pyplot as plt

class Line:
    """a: point on line
    d: direction of line (automatically normalised)."""
    def __init__(self, a, d):
        self.a = a.astype("float64")
        self.d = (d / np.linalg.norm(d)).astype("float64")

        self.perp_line = np.copy(self.d)
        self.perp_line[[0,1]] = self.perp_line[[1,0]]
        self.perp_line[0] *= -1
    
    def get(self, t):
        return self.a + t*self.d
    
def is_below_line(line, point):
    lp1 = line.get(0)
    lp2 = line.get(1)

    return ((lp2[0] - lp1[0])*(point[1] - lp1[1]) - (lp2[1] - lp1[1])*(point[0] - lp1[0]) < 0)[0] # "if below line"
    
def shoelace_area(points):
    lt = points.shape[1]
    return 0.5 * abs(sum([(points[1][i]+points[1][(i+1)%lt])*(points[0][i]-points[0][(i+1)%lt]) for i in range(lt)]))

def get_zero_crossings(y, wrap=False):
    signs_are_positive = (np.sign(y)>=0).astype(int)

    if wrap:
        upward_zero_crossings = np.where(np.roll(signs_are_positive, -1)-signs_are_positive == 1)[0]
        downward_zero_crossings = np.where(np.roll(signs_are_positive, -1)-signs_are_positive == -1)[0]

        for idx, i in enumerate(upward_zero_crossings):
            if abs(y[(i+1)%len(y)]) < abs(y[i]):
                upward_zero_crossings[idx] = (upward_zero_crossings[idx]+1)%len(y)
        for idx, i in enumerate(downward_zero_crossings):
            if abs(y[(i+1)%len(y)]) < abs(y[i]):
                downward_zero_crossings[idx] = (downward_zero_crossings[idx]+1)%len(y)
    else:
        upward_zero_crossings = np.where(np.diff(signs_are_positive) == 1)[0]
        downward_zero_crossings = np.where(np.diff(signs_are_positive) == -1)[0]

        for idx, i in enumerate(upward_zero_crossings[:-1]):
            if abs(y[i+1]) < abs(y[i]):
                upward_zero_crossings[idx] = upward_zero_crossings[idx]+1
        for idx, i in enumerate(downward_zero_crossings[:-1]):
            if abs(y[i+1]) < abs(y[i]):
                downward_zero_crossings[idx] = downward_zero_crossings[idx]+1

    return upward_zero_crossings, downward_zero_crossings

class Triangle:
    def __init__(self, points):
        if isinstance(points, np.ndarray):
            self.x = points
        else:
            self.x = np.array(points).T # numpy 2x3 array (2D, 3 points) with columns as the position vectors of each point

        self.centroid = self.x.mean(axis=1)[np.newaxis].T # numpy 2x1 array (2D, 1 point) as the position vector of centroid
        self.area = np.abs(0.5*np.linalg.det(np.hstack((self.x.T, np.array([[1,1,1]]).T)))) # area of triangle

        self.top = np.max(self.x[1]) # y coordinate of highest point on triangle.
        self.bottom = np.min(self.x[1]) # y coordinate of lowest point on triangle.

    # must go through three points of triangle, picking pairs of points, in a circle - otherwise the edge case of when the line intersects
    # a corner of the triangle will cause problems.
    def get_triangles_below(self, line):

        intersect_points = []
        for i in range(3):
            p_start = self.x[:,i][np.newaxis].T
            p_end = self.x[:,((i+1)%3)][np.newaxis].T
            l = Line(p_start, p_end - p_start)

            # line is intersecting Line. l is main Line, t=0 at one corner and t=t_end at the other.
            # if t_intersect is between t=0 and t=t_end (at point p_end), the point intersects the triangle. if not, it doesn't
            if not np.allclose(l.d, line.d) and not np.allclose(l.d, -line.d):
                try:
                    t_intersect = np.matmul(np.linalg.inv(np.hstack([l.d, -line.d])), line.a - l.a)[0]
                except:
                    print(f"line.a \n{line.a}\nline.d \n{line.d}\nl.a \n{l.a}\nl.d \n{l.d}\n" )
                    print(line.d.shape)
                    raise ValueError()
                if l.d[0] != 0:
                    t_end = (p_end[0] - l.a[0]) / l.d[0]
                else:
                    t_end = (p_end[1] - l.a[1]) / l.d[1]

                if t_intersect/t_end < 1 and t_intersect/t_end >= 0:
                    intersect_points.append(l.get(t_intersect))

        area_points = [] # points used to calculate area of triangle under line
        if len(intersect_points) == 2:
            for i in range(3):
                # check which side of line point i is on
                p = self.x[:,i][np.newaxis].T
                if is_below_line(line, p):
                    area_points.append(p)
            area_points = np.hstack(area_points + intersect_points)

        elif len(intersect_points) == 1 or len(intersect_points) == 0: # triangle is either entirely above or entirely below line
            for i in range(3):
                # check which side of line point i is on
                p = self.x[:,i][np.newaxis].T
                if is_below_line(line, p):
                    area_points = self.x
                    break

        if len(area_points) == 0: # if a is empty
            return []
        elif area_points.shape[1] == 3:
            # return np.abs(0.5*np.linalg.det(np.hstack((area_points.T, np.array([[1,1,1]]).T)))), k # area of triangle
            return [Triangle(area_points)]
        elif area_points.shape[1] == 4:
            area1 = shoelace_area(area_points)
            triangle_a_1 = Triangle((area_points[:,0], area_points[:,1], area_points[:,2]))
            triangle_b_1 = Triangle((area_points[:,0], area_points[:,2], area_points[:,3]))

            area_points[:, [0, 1]] = area_points[:, [1, 0]] # swap 0 and 1 columns of area_points
            area2 = shoelace_area(area_points)
            triangle_a_2 = Triangle((area_points[:,0], area_points[:,1], area_points[:,2]))
            triangle_b_2 = Triangle((area_points[:,0], area_points[:,2], area_points[:,3]))
            
            if area1 > area2:
                return [triangle_a_1, triangle_b_1]
            else:
                return [triangle_a_2, triangle_b_2]

class Shape:
    def __init__(self, triangles):
        self.triangles = triangles

        if len(triangles) == 0:
            self.centroids = np.array([[]])
        else:
            self.centroids = np.hstack([triangle.centroid for triangle in self.triangles])
        self.areas = np.array([triangle.area for triangle in self.triangles])

        self.centroid = (np.sum(self.centroids * self.areas, axis=1) / np.sum(self.areas))[np.newaxis].T
        self.area = np.sum(self.areas)

        if len(triangles) > 0:
            self.top = max([triangle.top for triangle in self.triangles]) # y coordinate of highest point on shape.
            self.bottom = min([triangle.bottom for triangle in self.triangles]) # y coordinate of lowest point on shape.

    def get_shape_below(self, line):
        triangles_below = []
        for triangle in self.triangles:
            triangles_below += triangle.get_triangles_below(line)
        return Shape(triangles_below)

    # theta is anticlockwise rotation of shape from origin (in radians)
    # returns the height relative to the water that the centroid floats at, and the area of the underwater section
    def float_height(self, theta, relative_density, brute_N=100):
        line = Line(self.centroid, np.array([[np.cos(theta),-np.sin(theta)]]).T) # start at centroid to avoid problems with flat gradient

        count=0
        dh = 1e-5
        h = 0
        diff = 1
        threshold = 1e-5
        while abs(diff) > threshold:
            shape = self.get_shape_below(Line(line.a+h*line.perp_line, line.d))
            shape2 = self.get_shape_below(Line(line.a+(h+dh)*line.perp_line, line.d))
            diff = shape.area / self.area - relative_density
            diff2 = shape2.area / self.area - relative_density

            gradient = (diff2 - diff) / dh
            if gradient == 0:
                print(f"Height gradient zero (iterative failed). Brute forcing, may take slightly longer. Theta(deg): {np.rad2deg(theta):.2f}, diff: {diff:.2f}, diff2: {diff2:.2f}, count: {count}")
                brute_h = np.linspace(self.bottom-self.centroid[1][0], self.top-self.centroid[1][0], brute_N)
                brute_diff = [self.get_shape_below(Line(line.a+x*line.perp_line, line.d)).area / self.area - relative_density for x in brute_h]
                i_min = np.concatenate(get_zero_crossings(brute_diff))[0]

                line = Line(line.a+brute_h[i_min]*line.perp_line, line.d)
                return -brute_h[i_min], self.get_shape_below(line), line

            h -= diff / gradient
            count += 1
            if count > 1000:
                raise ValueError("Max number of iterations (1000) reached when calculating float depth. Are you sure the object floats?")

        return -float(h), shape, line
    
    def float_angle_energy(self, relative_density, theta_0=0):
        count = 0
        theta = theta_0
        prev_theta = theta_0 + 1
        learning_rate = 0.1

        dtheta = 1e-5
        threshold = 1e-4

        # while abs(theta - prev_theta) > threshold:
        while abs(theta - prev_theta) > threshold or ggradient < 0:
            h, sub_shape, line = self.float_height(theta, relative_density)
            potential = (relative_density * self.area * 9.81 * h) -  (9.81 * sub_shape.area * (np.dot(sub_shape.centroid.T[0]-self.centroid.T[0], line.perp_line.T[0]) + h))
            h2, sub_shape2, line2 = self.float_height(theta+dtheta, relative_density)
            potential2 = (relative_density * self.area * 9.81 * h2) -  (9.81 * sub_shape2.area * (np.dot(sub_shape2.centroid.T[0]-self.centroid.T[0], line2.perp_line.T[0]) + h2))
            h3, sub_shape3, line3 = self.float_height(theta+2*dtheta, relative_density)
            potential3 = (relative_density * self.area * 9.81 * h3) -  (9.81 * sub_shape3.area * (np.dot(sub_shape3.centroid.T[0]-self.centroid.T[0], line3.perp_line.T[0]) + h3))

            gradient12 = (potential2 - potential) / dtheta
            gradient23 = (potential3 - potential2) / dtheta
            ggradient = (gradient23 - gradient12) / dtheta

            prev_theta = theta
            theta -= gradient12 * learning_rate

            count += 1
            if count % 50 == 0:
                print(count, np.rad2deg(theta))
            if count > 500:
                raise ValueError("Max number of iterations (500) reached when calculating theta.")
            
        return theta, h, sub_shape

    def float_angle_brute_energy(self, relative_density, N=100, plot=False):
        thetas = np.arange(-np.pi, np.pi, 1/N)
        potentials = np.zeros_like(thetas)
        heights = np.zeros_like(thetas)
        sub_shapes = []
        for i, theta in enumerate(thetas):
            h, sub_shape, line = self.float_height(theta, relative_density)
            potentials[i] = (relative_density * self.area * 9.81 * h) -  (9.81 * sub_shape.area * (np.dot(sub_shape.centroid.T[0]-self.centroid.T[0], line.perp_line.T[0]) + h))
            heights[i] = h
            sub_shapes.append(sub_shape)

        if plot:
            plt.plot(np.rad2deg(thetas), potentials)
            plt.xlabel("Theta")
            plt.ylabel("Potential")
            plt.show()

        i_min = np.argmin(potentials)
        return thetas[i_min], heights[i_min], sub_shapes[i_min]

    def float_angle_brute_gz(self, relative_density, N=20, plot=False):
        thetas = np.arange(-np.pi, np.pi, 1/N)
        righting_arms = np.zeros_like(thetas)
        heights = np.zeros_like(thetas)
        sub_shapes = []
        for i, theta in enumerate(thetas):
            h, sub_shape, line = self.float_height(theta, relative_density)
            righting_arms[i] = -np.dot(sub_shape.centroid.T[0]-self.centroid.T[0], line.d.T[0])
            heights[i] = h
            sub_shapes.append(sub_shape)

        if plot:
            plt.plot(np.rad2deg(thetas), righting_arms)
            plt.xlabel("Theta")
            plt.ylabel("GZ Righting arm")
            plt.grid()
            plt.show()

        # zero crossings of righting arm with +ve gradient are equilibria, zero crossings with -ve gradient are points of vanishing stability
        upward_zero_crossings = get_zero_crossings(righting_arms, wrap=True)[0]

        # returns an arbitrarily chosen equilibrium
        i_min = upward_zero_crossings[-1]

        return thetas[i_min], heights[i_min], sub_shapes[i_min]

    def catastrophe_locus(self, relative_density, N=20, theta_range=None, plot=False):
        if theta_range is None:
            theta_range = [-np.pi*1/5, np.pi*1/5]
            # theta_range = [-np.pi, np.pi]

        buoyancy_points = [] # buoyancy locus
        perp_locus_lines = [] # perpendicular lines from buoyancy locus
        evolute_points = [] # evolute of buoyancy locus
        thetas = np.arange(theta_range[0], theta_range[1], 1/N)

        dtheta = 1e-5
        for idx, theta in enumerate(thetas):
            height, submerged_shape, line = self.float_height(theta, relative_density)
            buoyancy_points.append(submerged_shape.centroid)
            
            # get line perpendicular to buoyancy locus
            buoyancy_point_right = self.float_height(theta+dtheta, relative_density)[1].centroid
            perp_locus_lines.append(Line(submerged_shape.centroid, Line(submerged_shape.centroid, submerged_shape.centroid-buoyancy_point_right).perp_line))

            if idx > 0:
                t_intersect = np.matmul(np.linalg.inv(np.hstack([perp_locus_lines[idx].d, -perp_locus_lines[idx-1].d])), perp_locus_lines[idx-1].a - perp_locus_lines[idx].a)[0]
                point_intersect = perp_locus_lines[idx].a + t_intersect*perp_locus_lines[idx].d
                # print(t_intersect)
                evolute_points.append(point_intersect)
            
        if plot:
            plt.scatter(*zip(*buoyancy_points), c="black", s=5, label="buoyancy locus")
            plt.scatter(*zip(*evolute_points), c="blue", s=5, label="evolute locus")
            plt.legend()
            plt.grid()
            plt.show()

        return buoyancy_points, evolute_points, perp_locus_lines

square_tr_bottom = Triangle(((-1,-1),(1,-1),(1,1)))
square_tr_top = Triangle(((-1,-1),(-1,1),(1,1)))
square = Shape((square_tr_bottom, square_tr_top))

# Thin anchor
# tr1 = Triangle(((0,0),(2,1),(0,-1)))
# tr2 = Triangle(((0,0),(-2,1),(0,-1)))
# tr3 = Triangle(((0,0),(0.5,0.25),(0,5)))
# tr4 = Triangle(((0,0),(-0.5,0.25),(0,5)))
# square = Shape((tr1,tr2,tr3,tr4))

square.catastrophe_locus(0.22, N=100, plot=True)
# square.float_angle_brute_gz(0.1, plot=True)
# print(np.rad2deg(square.float_angle_brute_energy(0.1, plot=True)[0]))

#PLOT2
#----
# densities = np.linspace(0, 1, 102)[1:-1]
# thetas = []
# for idx, d in enumerate(densities):
#     thetas.append(square.float_angle_brute_gz(d, N=100)[0])
#     print(f"{100*(idx+1)/len(densities):.2f}%")
# plt.plot(densities, np.rad2deg(np.array(thetas)))
# plt.ylabel("Theta (degrees)")
# plt.xlabel("Relative Density")
# plt.grid()
# plt.show()
#----

#PLOT1
# ----
# nd = 101 # number densities
# nt = 51 # number thetas
# densities = np.linspace(0, 1, nd+2)[1:-1]
# t_start, t_end = -np.pi/4, 0
# thetas = np.linspace(t_start, t_end, nt)
# optimal_thetas = []
# potentials = np.zeros(shape=(nd,nt))
# for d, density in enumerate(densities):
#     for t, theta in enumerate(thetas):
#         h, sub_shape, line = square.float_height(theta, density)
#         potentials[d][t] = (density * square.area * 9.81 * h) -  (9.81 * sub_shape.area * (np.dot(sub_shape.centroid.T[0]-square.centroid.T[0], line.perp_line.T[0]) + h))
#     potentials[d] -= np.min(potentials[d])
#     optimal_thetas.append(thetas[np.argmin(potentials[d])])
#     print(f"{d/nd * 100:.2f}%")

# plt.plot(densities, np.rad2deg(optimal_thetas))
# plt.xlabel("Relative density")
# plt.ylabel("Angle of minimum potential")
# plt.grid()
# plt.show()

# fig = plt.figure(figsize=(13, 7))
# ax = plt.axes(projection='3d')
# xx, yy = np.mgrid[0:1:nd*1j, t_start:t_end:nt*1j]
# surf = ax.plot_surface(xx, yy, potentials, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
# ax.set_xlabel('Relative Density')
# ax.set_ylabel('Angle')
# ax.set_zlabel("Potential")
# ax.view_init(60, 35)
# plt.show()
#----