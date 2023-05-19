import numpy as np
import pygame
import os, sys

from polygons import Line, Triangle, Shape

def vec2tuple(vec, do_round=True, offset=(0,0)):
    if do_round:
        return (int(round(vec[0][0] + offset[0])), int(round(vec[1][0] + offset[1])))

# Square
# square_tr_bottom = Triangle(((-1,-1),(1,-1),(1,1)))
# square_tr_top = Triangle(((-1,-1),(-1,1),(1,1)))
# square = Shape((square_tr_bottom, square_tr_top))

# Anchor
# tr1 = Triangle(((0,0),(2,1),(0,-1)))
# tr2 = Triangle(((0,0),(-2,1),(0,-1)))
# tr3 = Triangle(((0,0),(1,0.5),(0,3)))
# tr4 = Triangle(((0,0),(-1,0.5),(0,3)))
# square = Shape((tr1,tr2,tr3,tr4))

# Thin anchor
tr1 = Triangle(((0,0),(2,1),(0,-1)))
tr2 = Triangle(((0,0),(-2,1),(0,-1)))
tr3 = Triangle(((0,0),(0.5,0.25),(0,5)))
tr4 = Triangle(((0,0),(-0.5,0.25),(0,5)))
square = Shape((tr1,tr2,tr3,tr4))

# angle, height, submerged_shape = square.float_angle_brute_energy(0.25, N=20)
angle, height, submerged_shape = square.float_angle_brute_gz(0.25, N=40, plot=True)
print(np.rad2deg(angle))

# angle = np.pi/4
rotation_matrix_2d = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])

pygame.init()
size = WIDTH, HEIGHT = 960,540
screen = pygame.display.set_mode(size)
font = pygame.font.SysFont("arial", 20)

directory = os.getcwd()
IMG_centre_of_mass = pygame.transform.scale(pygame.image.load(f"{directory}\\assets\\CoM.png").convert_alpha(), (30,30))
IMG_centre_of_buoyancy = pygame.transform.scale(pygame.image.load(f"{directory}\\assets\\CoB.png").convert_alpha(), (30,30))

water_centre = np.array([[WIDTH*1/2, HEIGHT*3/5]]).T
px_per_m = 50

centroid_coord = px_per_m*np.array([[0,-height]]).T + water_centre
submerged_centroid_pos = np.dot(rotation_matrix_2d, submerged_shape.centroid - square.centroid) + np.array([[0,height]]).T
submerged_centroid_coord = px_per_m*np.array([[1,-1]]).T*submerged_centroid_pos + water_centre

text_height = font.render(f"Centroid float height: {height:.2f}m", True, (255, 255, 255))
text_angle = font.render(f"Float angle: {np.rad2deg(angle):.2f} degrees", True, (255, 255, 255))
text_cobpos = font.render(f"Centre of buoyancy position: ({submerged_centroid_pos[0][0]:.2f}, {submerged_centroid_pos[1][0]:.2f})m", True, (255, 255, 255))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            sys.exit()

    screen.fill([135, 206, 235])
    pygame.draw.rect(screen, (15, 82, 186), pygame.Rect(0, HEIGHT*3/5, WIDTH, HEIGHT*2/5))

    for triangle in square.triangles:
        # print(square.centroid, triangle.x)
        point_relative_to_centroid = np.subtract(triangle.x, square.centroid)
        point_rotated = np.matmul(rotation_matrix_2d, point_relative_to_centroid)
        coord_draw = px_per_m*np.add(point_rotated, np.array([[0,height]]).T)
        coord_draw[1] = -coord_draw[1]
        coord_draw = np.add(coord_draw, water_centre)

        points = tuple(map(tuple, coord_draw.T))
        pygame.draw.polygon(screen, (66, 0, 105), points)
        pygame.draw.line(screen, (100,100,100), points[0], points[1])
        pygame.draw.line(screen, (100,100,100), points[1], points[2])
        pygame.draw.line(screen, (100,100,100), points[2], points[0])

    screen.blit(IMG_centre_of_mass, IMG_centre_of_mass.get_rect(center=vec2tuple(centroid_coord)))
    screen.blit(IMG_centre_of_buoyancy, IMG_centre_of_buoyancy.get_rect(center=vec2tuple(submerged_centroid_coord)))
    
    screen.blit(text_height, (7,HEIGHT-(32+40)))
    screen.blit(text_angle, (7,HEIGHT-(32+20))) 
    screen.blit(text_cobpos, (7,HEIGHT-32))    

    pygame.display.flip()