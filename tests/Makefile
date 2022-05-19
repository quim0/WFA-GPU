SM=80
COMPUTE=$(SM)
CC=gcc
WFAGPU_PATH=../

tests: api-test

api-test:
	$(CC) test_api.c -o test-api -I $(WFAGPU_PATH)/lib/ -I $(WFAGPU_PATH) -L $(WFAGPU_PATH)/build/ -L $(WFAGPU_PATH)/external/WFA/lib/ -lwfagpu -lwfa

run-tests:
	LD_LIBRARY_PATH=../build/ ./test-api
