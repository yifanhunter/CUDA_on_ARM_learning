TEST_SOURCE = Index_of_thread.cu

TARGETBIN := ./Index_of_thread

CC = /usr/local/cuda/bin/nvcc

$(TARGETBIN):$(TEST_SOURCE)
	$(CC)  $(TEST_SOURCE) -o $(TARGETBIN)

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)
