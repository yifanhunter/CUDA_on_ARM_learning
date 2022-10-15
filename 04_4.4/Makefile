TEST_SOURCE = sum.cu

TARGETBIN := ./sum

CC = /usr/local/cuda/bin/nvcc

$(TARGETBIN):$(TEST_SOURCE)
	$(CC)  $(TEST_SOURCE) -o $(TARGETBIN)

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)
