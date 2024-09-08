import math

d = 5.0
e = 7.937
f = 4.845
g = 8.72
hz = 12.5

def caucluate_a(ny, nz):
    c = ((e*ny)/math.sqrt(math.pow(nz, 2) + math.pow(ny, 2))) + hz
    b = -math.sqrt(math.pow(e, 2) - math.pow(c - hz, 2))

    I = (((b+d)/c)**2) + 1
    J = (((g**2)*(b+d))/(c**2)) - (b+d) + (2*d)
    K = ((c**2)/4) + ((g**4)/(4*(c**2))) - ((g**2)/2) - (f**2) + (d**2)

    Y = (-J - math.sqrt((J**2) - (4*I*K)))/(2*I)
    Z = math.sqrt((f**2) - ((Y + d)**2))

    theta_a = math.atan((-Z)/(Y + d))
    return(theta_a)

def calculate_b(nx, ny, nz):
    c = ((e*((nx*math.sqrt(3)) + ny))/math.sqrt((nz**2) + (((nx*math.sqrt(3)) + ny)**2))) + hz
    a = math.sqrt(3*((e**2) - ((c - hz)**2)))
    b = a/math.sqrt(3)

    I = ((((4*d)/(math.sqrt(3))) - (2*a) - ((2*b)/(math.sqrt(3))))**2) + ((16*(c**2))/3)
    J = (2*(((4*d)/(math.sqrt(3))) - (2*a) - ((2*b)/(math.sqrt(3))))*((a**2) + (b**2) - (d**2) + (f**2) + (c**2) - (g**2))) - ((16*d*(c**2))/(math.sqrt(3)))
    K = (((a**2) + (b**2) - (d**2) + (f**2) + (c**2) - (g**2))**2) + (4*(d**2)*(c**2)) - (4*(f**2)*(c**2))

    X = (-J + math.sqrt((J**2) - (4*I*K)))/(2*I)
    Y = X/math.sqrt(3)
    Z = (((X*4*d)/(math.sqrt(3))) - (d**2) + (f**2) - ((4*(X**2))/(3)))**(0.5)

    theta_b = math.atan(Z/((2*Y) - d))
    return(theta_b)

def calculate_c(nx, ny, nz):
    c = ((e*(ny - (math.sqrt(3)*nx)))/(math.sqrt((4*(nz**2)) + ((ny - (math.sqrt(3)*nx))**2)))) + hz
    a = -math.sqrt(3*((e**2) - ((hz-c)**2)))/2
    b = -a/math.sqrt(3)

    I = ((((4*d)/(math.sqrt(3))) + (2*a) + ((2*b)/(math.sqrt(3))))**2) + ((16*(c**2))/3)
    J = (2*(((4*d)/(math.sqrt(3))) + (2*a) + ((2*b)/(math.sqrt(3))))*((a**2) + (b**2) - (d**2) + (f**2) + (c**2) - (g**2))) + ((16*d*(c**2))/(math.sqrt(3)))
    K = (((a**2) + (b**2) - (d**2) + (f**2) + (c**2) - (g**2))**2) + (4*(d**2)*(c**2)) - (4*(f**2)*(c**2))

    X = (J - math.sqrt((J**2) - (4*I*K)))/(2*I)
    Y = -X/math.sqrt(3)
    Z = math.sqrt((f**2) - (d**2) - ((4*(X**2))/(3)) - ((X*4*d)/(math.sqrt(3))))

    theta_c = math.atan((Z)/((2*Y) - d))
    return(theta_c)