CXX=g++
CXXFLAGS= -pthread
NVCC=nvcc
# cuda location C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3
BIN=bin
ERA_SRC=Eratosthenes_sieve
SUN_SRC=Sundaram_sieve

all:
	@echo "Please use 'make <target>'"
	@echo "  compile1:		compile eratosthenes cpu"
	@echo "  compile2:		compile eratosthenes gpu"
	@echo "  compile3:		compile sundaram cpu"
	@echo "  compile4:		compile sundaram gpu"
	@echo "  run1:			run eratosthenes cpu"
	@echo "  run2:			run eratosthenes gpu"
	@echo "  run3:			run sundaram cpu"
	@echo "  run4:			run sundaram gpu"
	@echo "  clean:		remove bin folder"

create:
	if [ ! -d $(BIN) ]; then mkdir $(BIN); fi

compile1:
	$(CXX) -o $(BIN)/$(ERA_SRC)_cpu $(ERA_SRC)/CPU/main.cpp $(CXXFLAGS)

compile2:
	$(NVCC) -o $(BIN)/$(ERA_SRC)_gpu $(ERA_SRC)/GPU/main.cu

compile3:
	$(CXX) -o $(BIN)/$(SUN_SRC)_cpu $(SUN_SRC)/CPU/main.cpp $(CXXFLAGS)

compile4:
	$(NVCC) -o $(BIN)/$(SUN_SRC)_gpu $(SUN_SRC)/GPU/main.cu

run1: compile1
	./$(BIN)/$(ERA_SRC)_cpu

run2: compile2
	./$(BIN)/$(ERA_SRC)_gpu

run3: compile3
	./$(BIN)/$(SUN_SRC)_cpu

run4: compile4
	./$(BIN)/$(SUN_SRC)_gpu

clean:
	rm $(BIN) -rf