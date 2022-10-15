TEST_SOURCE = matrix_mul.cu

TARGETBIN := ./matrix_mul

CC = /usr/local/cuda/bin/nvcc

$(TARGETBIN):$(TEST_SOURCE)
	$(CC)  $(TEST_SOURCE) -o $(TARGETBIN)

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)
