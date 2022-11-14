TEST_SOURCE = matrix_mul.cu

TARGETBIN := ./matrix_mul

CC = /usr/local/cuda/bin/nvcc

$(TARGETBIN):$(TEST_SOURCE)
	$(CC) -arch=compute_80 -code=sm_80  $(TEST_SOURCE) -o $(TARGETBIN)

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)
