from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendbuf = None
if rank == 0:
    sendbuf = np.empty([size, 100], dtype='i')
    sendbuf.T[:,:] = range(size)
recvbuf = np.empty(100, dtype='i')
comm.Scatter(sendbuf, recvbuf, root=0)

print("Process", rank, "has data:", recvbuf, "|||",  len(recvbuf))


#
# # Define the two polynomials as arrays of coefficients
# poly1 = [1, 0, 0]
# poly2 = [1, 0, 0]
#
# # Find the degree of the polynomials
# degree1 = len(poly1) - 1
# degree2 = len(poly2) - 1
#
# # Create a new array to store the result of the multiplication
# result = np.zeros(degree1 + degree2 + 1)
#
# # Scatter the polynomials across the different processes
# if rank == 0:
#     data1 = np.array_split(poly1, size)
#     data2 = np.array_split(poly2, size)
# else:
#     data1 = None
#     data2 = None
#
# local_poly1 = comm.scatter(data1, root=0)
# local_poly2 = comm.scatter(data2, root=0)
#
# # Perform the multiplication on each process
# local_result = np.polymul(local_poly1, local_poly2)
#
# # Collect the results from all the processes
# comm.Reduce(local_result, result, op=MPI.SUM, root=0)
#
# if rank == 0:
#     print("Result:", result)
