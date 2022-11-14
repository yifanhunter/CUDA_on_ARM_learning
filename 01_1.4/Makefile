TEST_SOURCE = hello_cuda.cu

TARGETBIN := ./hello_cuda

CC = /usr/local/cuda/bin/nvcc


$(TARGETBIN):$(TEST_SOURCE)
	$(CC)  -arch=compute_80 -code=sm_80 $(TEST_SOURCE) -o $(TARGETBIN)

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)
	-rm -rf *.o
